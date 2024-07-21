import argparse

from deepface import DeepFace
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import numpy as np
import os
from typing import List, Optional, Dict
import tqdm
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Wav2Vec2Processor
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: List,
            basedir: Optional[str] = None,
            sampling_rate: int = 16000,
            max_audio_len: int = 5,
    ):
        self.dataset = dataset
        self.basedir = basedir
        self.sampling_rate = sampling_rate
        self.max_audio_len = max_audio_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        filepath = self.dataset[index]
        speech_array, sr = torchaudio.load(filepath)

        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        if sr != self.sampling_rate:
            transform = torchaudio.transforms.Resample(sr, self.sampling_rate)
            speech_array = transform(speech_array)
            sr = self.sampling_rate

        len_audio = speech_array.shape[1]

        if len_audio < self.max_audio_len * self.sampling_rate:
            padding = torch.zeros(1, self.max_audio_len * self.sampling_rate - len_audio)
            speech_array = torch.cat([speech_array, padding], dim=1)
        else:
            speech_array = speech_array[:, :self.max_audio_len * self.sampling_rate]

        speech_array = speech_array.squeeze().numpy()
        return {"input_values": speech_array, "attention_mask": None}


class CollateFunc:
    def __init__(
            self,
            processor: Wav2Vec2Processor,
            padding: bool = True,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: bool = True,
            sampling_rate: int = 16000,
            max_length: Optional[int] = None,
    ):
        self.sampling_rate = sampling_rate
        self.processor = processor
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_attention_mask = return_attention_mask
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, np.ndarray]]):
        input_values = [item["input_values"] for item in batch]
        batch = self.processor(
            input_values,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask
        )
        return {
            "input_values": batch.input_values,
            "attention_mask": batch.attention_mask if self.return_attention_mask else None
        }


def predict(test_dataloader, model, device: torch.device):
    model.to(device)
    model.eval()
    scores_list = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader):
            input_values, attention_mask = batch['input_values'].to(device), batch['attention_mask'].to(device)
            logits = model(input_values, attention_mask=attention_mask).logits
            scores = F.softmax(logits, dim=-1)
            scores_list.extend(scores.cpu().detach().numpy())
    return scores_list


def get_audio_paths(base_dir):
    audio_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_paths.append(os.path.join(root, file))
    return audio_paths


def run_gender_recognition(base_dir):
    audio_paths = get_audio_paths(base_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label2id = {
        "female": 0,
        "male": 1
    }

    id2label = {
        0: "female",
        1: "male"
    }

    model_name_or_path = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"

    # Preload model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    model = AutoModelForAudioClassification.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    test_dataset = CustomDataset(audio_paths, max_audio_len=5)
    data_collator = CollateFunc(
        processor=feature_extractor,
        padding=True,
        sampling_rate=16000,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=2
    )

    scores = predict(test_dataloader=test_dataloader, model=model, device=device)

    with open("results.txt", "w") as f:
        for path, score in zip(audio_paths, scores):
            score_str = " ".join(map(str, score))
            relative_path = os.path.relpath(path, base_dir)
            f.write(f"{relative_path}  {score_str}\n")

    print("Results have been written to results.txt")


def get_image_paths(base_dir):
    image_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    return image_paths


def facial_analysis(image_path, detector):
    try:
        obj = DeepFace.analyze(img_path=image_path, actions=['gender', 'age'], detector_backend=detector)
    except Exception as e:
        print(f"Error during analysis: {e}")
        obj = DeepFace.analyze(img_path=image_path, actions=['gender', 'age'], detector_backend=detector,
                               enforce_detection=False)

    if 'gender' in obj[0] and 'age' in obj[0]:
        return obj[0]['gender'], obj[0]['age']
    else:
        raise ValueError("Gender or age information not found in the analysis result")


def process_files(input_file_path, result_file_path, output_file_path):
    u_id_wav = []
    u_id_jpg = []

    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                u_id_wav.append(parts[0])
                u_id_jpg.append(parts[1])

    f_g = []
    f_a = []
    f_id = []

    with open(result_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                f_id.append(parts[0])
                f_g.append(parts[1])
                f_a.append(parts[3])

    output_content = []

    for u_id in u_id_jpg:
        if u_id in f_id:
            index = f_id.index(u_id)
            output_content.append(f"{f_a[index]} {f_g[index]}")

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in output_content:
            file.write(line + '\n')

    print(f"File written to {output_file_path}")


def process_wav_files(input_file_path, result_file_path, output_file_path):
    u_id_wav = []

    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                u_id_wav.append(parts[0])

    w_g = []
    w_id = []
    output_content = []

    with open(result_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                w_id.append(parts[0])
                w_g.append((parts[1].split())[0])

    for u_id in u_id_wav:
        if u_id in w_id:
            index = w_id.index(u_id)
            output_content.append(f"{w_g[index]}")

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in output_content:
            file.write(line + '\n')

    print(f"File written to {output_file_path}")


class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


def initialize_model_and_processor():
    global device, processor, model, sampling_rate
    device = 'cpu'
    model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = AgeGenderModel.from_pretrained(model_name)
    sampling_rate = 16000


def process_func(x: np.ndarray, sampling_rate: int, embeddings: bool = False) -> np.ndarray:
    r"""Predict age and gender or extract embeddings from raw audio signal."""
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)
    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0]
        else:
            y = torch.hstack([y[1], y[2]])
    y = y.detach().cpu().numpy()
    return y


def read_wav_file(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != sampling_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sampling_rate)
        waveform = transform(waveform)
    return waveform.squeeze().numpy()


def process_directory(root_dir, output_file):
    with open(output_file, 'w') as out_file:
        for id_folder in os.listdir(root_dir):
            id_folder_path = os.path.join(root_dir, id_folder)
            if os.path.isdir(id_folder_path):
                for lang_folder in ['Urdu', 'English']:
                    lang_folder_path = os.path.join(id_folder_path, lang_folder)
                    if os.path.isdir(lang_folder_path):
                        for sub_folder in os.listdir(lang_folder_path):
                            sub_folder_path = os.path.join(lang_folder_path, sub_folder)
                            if os.path.isdir(sub_folder_path):
                                for wav_file in os.listdir(sub_folder_path):
                                    if wav_file.endswith('.wav'):
                                        wav_file_path = os.path.join(sub_folder_path, wav_file)
                                        signal = read_wav_file(wav_file_path)
                                        signal = signal.reshape(1, -1)
                                        result = process_func(signal, sampling_rate)
                                        relative_path = os.path.relpath(wav_file_path, root_dir)
                                        out_file.write(f"{relative_path}  {result[0][0]}\n")
                                        print(f"Processed: {relative_path} {result[0][0]}")


def process_and_append_files(wav_e_a_path, e_rest_wav_path):
    processed_numbers = []
    with open(wav_e_a_path, 'r', encoding='utf-8') as file:
        for line in file:
            number = float(line.strip())
            processed_number = number * 100
            processed_numbers.append(f"{processed_number}")

    with open(e_rest_wav_path, 'r', encoding='utf-8') as file:
        original_lines = file.readlines()

    if len(original_lines) != len(processed_numbers):
        raise ValueError(f"{wav_e_a_path} and {e_rest_wav_path} do not have the same number of lines")

    with open(e_rest_wav_path, 'w', encoding='utf-8') as file:
        for i in range(len(original_lines)):
            new_line = f"{processed_numbers[i]} {original_lines[i].strip()}\n"
            file.write(new_line)

    print(f"File written to {e_rest_wav_path}")


def generate_results_ag_face(base_dir, detector_backend):
    image_paths = get_image_paths(base_dir)
    output_file = 'results_ag_face.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_path in image_paths:
            try:
                gender, age = facial_analysis(image_path, detector_backend)
                gender_woman = gender['Woman']
                gender_man = gender['Man']
                relative_path = os.path.relpath(image_path, base_dir)
                f.write(f"{relative_path} {gender_woman} {gender_man} {age}\n")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    print(f"Results have been written to {output_file}")


def process_wav_a_files(input_file_path, result_file_path, output_file_path):
    u_id_wav = []

    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                u_id_wav.append(parts[0])

    w_id = []
    w_a = []
    output_content = []

    with open(result_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                w_id.append(parts[0])
                w_a.append(parts[1])

    for u_id in u_id_wav:
        if u_id in w_id:
            index = w_id.index(u_id)
            output_content.append(f"{w_a[index]}")

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in output_content:
            file.write(line + '\n')

    print(f"File written to {output_file_path}")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_age_confidence(age_audio, age_face):
    return 1 / (1 + np.abs(age_audio - age_face))


def calculate_gender_confidence(gender_prob_audio, gender_prob_face):
    return gender_prob_audio * gender_prob_face + (1 - gender_prob_audio) * (1 - gender_prob_face)


def calculate_overall_confidence(age_confidence, gender_confidence, w_a=0.5, w_g=0.5):
    return w_a * age_confidence + w_g * gender_confidence


def adjust_l2_score(initial_score, confidence, threshold, alpha):
    if confidence > threshold:
        return initial_score / alpha
    else:
        return initial_score * alpha


def age_gender_l2_adjustment(age_audio, age_face, gender_score_audio, gender_score_face, initial_score, threshold=0.5,
                             alpha=1.2, w_a=0.5, w_g=0.5):
    age_confidence = calculate_age_confidence(age_audio, age_face)
    gender_confidence = calculate_gender_confidence(gender_score_audio, gender_score_face)
    # overall_confidence = calculate_overall_confidence(age_confidence, gender_confidence, w_a, w_g)
    # adjusted_score = adjust_l2_score(initial_score, overall_confidence, threshold, alpha)

    return age_confidence, gender_confidence


# # 定义文件路径
def process_files_fianl(e_rest_wav_path, e_rest_face_path, e_con_path, output_path):
    # 初始化列表
    age_audio = []
    age_face = []
    gender_score_audio = []
    gender_score_face = []

    # 读取e_rest_wav.txt文件
    with open(e_rest_wav_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                age_audio.append(float(parts[0]))
                gender_score_audio.append(float(parts[1]))

    # 读取e_rest_face.txt文件并将值转换为浮点数
    with open(e_rest_face_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                age_face.append(float(parts[0]))
                gender_score_face.append(float(parts[1]) / 100)

    # 打印结果以进行调试
    with open(e_con_path, 'w', encoding='utf-8') as file:
        for i in range(len(age_audio)):
            age_conf, gender_conf = age_gender_l2_adjustment(
                age_audio[i], age_face[i], gender_score_audio[i], gender_score_face[i], initial_score=0
            )
            file.write(f"{age_conf} {gender_conf}\n")

    # 打开输入文件和输出文件
    with open(e_con_path, 'r') as infile, open(output_path, 'w') as outfile:
        # 逐行读取输入文件
        for line in infile:
            # 分割每行的两个浮点数
            num1, num2 = map(float, line.split())
            # 计算它们的和
            sum_nums = num1 + num2
            # 判断和是否大于1.11，并写入输出文件
            if sum_nums > 1:
                outfile.write('1\n')
            else:
                outfile.write('0\n')

    print(f"处理完成，结果已写入 {output_path} 文件。")


def adjust_scores(e_output_lines, heard_file, unheard_file):
    # 读取 heard_file 的内容
    with open(heard_file, 'r') as heard:
        heard_lines = heard.readlines()

    # 读取 unheard_file 的内容
    with open(unheard_file, 'r') as unheard:
        unheard_lines = unheard.readlines()

    # 检查文件行数是否匹配
    if not (len(e_output_lines) == len(heard_lines) == len(unheard_lines)):
        print(f"错误: 文件行数不匹配 ({heard_file}, {unheard_file})")
        return

    # 调整分数并写回文件
    adjusted_heard_lines = []
    adjusted_unheard_lines = []

    for i in range(len(e_output_lines)):
        e_output_value = e_output_lines[i]

        heard_parts = heard_lines[i].strip().split()
        unheard_parts = unheard_lines[i].strip().split()

        heard_id, heard_score = heard_parts[0], float(heard_parts[1])
        unheard_id, unheard_score = unheard_parts[0], float(unheard_parts[1])

        if e_output_value == 1:
            heard_score /= 1.2
            unheard_score /= 1.2
        else:
            heard_score *= 1.2
            unheard_score *= 1.2

        adjusted_heard_lines.append(f"{heard_id} {heard_score}\n")
        adjusted_unheard_lines.append(f"{unheard_id} {unheard_score}\n")

    # 写回调整后的内容
    with open(heard_file, 'w') as heard:
        heard.writelines(adjusted_heard_lines)

    with open(unheard_file, 'w') as unheard:
        unheard.writelines(adjusted_unheard_lines)

    print(f"分数调整完成，结果已写入文件: {heard_file}, {unheard_file}")


def main(base_dir, root_dir, folders):
    datasets = ['Urdu', 'English']

    detector_backend = 'mtcnn'
    initialize_model_and_processor()

    output_file = 'results_a_wav.txt'
    run_gender_recognition(root_dir)
    process_directory(root_dir, output_file)
    generate_results_ag_face(base_dir, detector_backend)

    for dataset in datasets:
        input_file_path = f'{dataset}_test_final.txt'
        result_file_path = 'results_ag_face.txt'
        output_file_path = f'result_face_{dataset.lower()}.txt'

        process_files(input_file_path, result_file_path, output_file_path)

        result_file_path = 'results_g_wav.txt'
        output_file_path = f'result_wav_{dataset.lower()}.txt'

        process_wav_files(input_file_path, result_file_path, output_file_path)

        result_file_path = 'results_a_wav.txt'
        output_file_path = f'wav_{dataset.lower()}_a.txt'

        process_wav_a_files(input_file_path, result_file_path, output_file_path)

        wav_e_a_path = f'wav_{dataset.lower()}_a.txt'
        e_rest_wav_path = f'result_wav_{dataset.lower()}.txt'

        process_and_append_files(wav_e_a_path, e_rest_wav_path)
        os.remove(wav_e_a_path)
        print(f"Temporary file {wav_e_a_path} deleted.")
    english_files = {
        'wav_path': 'result_wav_english.txt',
        'face_path': 'result_face_english.txt',
        'con_path': 'english_con.txt',
        'output_path': 'e_output.txt'
    }

    urdu_files = {
        'wav_path': 'result_wav_urdu.txt',
        'face_path': 'result_face_urdu.txt',
        'con_path': 'urdu_con.txt',
        'output_path': 'u_output.txt'
    }

    # 处理英语文件
    process_files_fianl(english_files['wav_path'], english_files['face_path'], english_files['con_path'],
                        english_files['output_path'])
    # 处理乌尔都语文件
    process_files_fianl(urdu_files['wav_path'], urdu_files['face_path'], urdu_files['con_path'],
                        urdu_files['output_path'])

    # 删除中间生成的con文件
    os.remove(english_files['con_path'])
    os.remove(urdu_files['con_path'])
    os.remove(english_files['wav_path'])
    os.remove(english_files['face_path'])
    os.remove(urdu_files['wav_path'])
    os.remove(urdu_files['face_path'])

    with open('e_output.txt', 'r') as e_output_file:
        e_output_lines = [int(line.strip()) for line in e_output_file]

    # 读取 u_output.txt 的内容
    with open('u_output.txt', 'r') as u_output_file:
        u_output_lines = [int(line.strip()) for line in u_output_file]

    # 定义文件夹和文件名

    heard_file_template = 'sub_score_{}_heard.txt'
    unheard_file_template = 'sub_score_{}_unheard.txt'

    # 遍历每个文件夹
    for folder in folders:
        # 调整 English 分数
        heard_file = os.path.join(folder, heard_file_template.format('English'))
        unheard_file = os.path.join(folder, unheard_file_template.format('English'))

        # 检查文件是否存在
        if not os.path.exists(heard_file) or not os.path.exists(unheard_file):
            print(f"文件 {heard_file} 或 {unheard_file} 不存在，跳过...")
        else:
            adjust_scores(e_output_lines, heard_file, unheard_file)

        # 调整 Urdu 分数
        heard_file = os.path.join(folder, heard_file_template.format('Urdu'))
        unheard_file = os.path.join(folder, unheard_file_template.format('Urdu'))

        # 检查文件是否存在
        if not os.path.exists(heard_file) or not os.path.exists(unheard_file):
            print(f"文件 {heard_file} 或 {unheard_file} 不存在，跳过...")
        else:
            adjust_scores(u_output_lines, heard_file, unheard_file)

        os.remove('e_output.txt')
        os.remove('u_output.txt')
        os.remove('results_ag_face.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process paths for face images, audio files, and score folders.")
    parser.add_argument('face_image_path', type=str, help='Path to the face images')
    parser.add_argument('audio_path', type=str, help='Path to the audio files')
    parser.add_argument('score_folders', nargs='+', help='List of score folders')

    args = parser.parse_args()

    main(args.face_image_path, args.audio_path, args.score_folders)
