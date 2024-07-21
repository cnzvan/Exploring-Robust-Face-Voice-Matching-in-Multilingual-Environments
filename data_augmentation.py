import random
import csv
import argparse


def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return [row for row in reader]


def process_files(input_prefix, output_prefix, factor):
    reset_file = f'{input_prefix}_train_filtered_reset.txt'
    voices_file = f'{input_prefix}_voices_train_filtered.csv'
    faces_file = f'{input_prefix}_faces_train_filtered.csv'

    reset_lines = read_file(reset_file)
    voices_lines = read_csv(voices_file)
    faces_lines = read_csv(faces_file)

    assert len(reset_lines) == len(voices_lines) == len(faces_lines), "All files must have the same number of lines"

    count = int(len(reset_lines) * factor)

    id_to_lines = {}
    for line in reset_lines:
        id_prefix = line.split('/')[0]
        if id_prefix not in id_to_lines:
            id_to_lines[id_prefix] = []
        id_to_lines[id_prefix].append(line)

    correct_correspondence = set()
    correct_voices = []
    correct_faces = []

    all_ids = list(id_to_lines.keys())
    random.shuffle(all_ids)

    used_indices = set()

    while len(correct_correspondence) < count:
        id_prefix = random.choice(all_ids)
        lines = id_to_lines[id_prefix]

        part1 = random.choice(lines)
        part2 = random.choice(lines)

        parts1 = part1.split()
        parts2 = part2.split()

        voice_indices = [i for i, l in enumerate(reset_lines) if l == part1]
        face_indices = [i for i, l in enumerate(reset_lines) if l == part2]

        if not voice_indices or not face_indices:
            print(f"Warning: Not enough indices found for {part1} or {part2}. Skipping this iteration.")
            continue

        voice_idx = random.choice(voice_indices)
        face_idx = random.choice(face_indices)

        indices_tuple = (voice_idx, face_idx)

        if indices_tuple in used_indices:
            continue

        used_indices.add(indices_tuple)

        voice_line = voices_lines[voice_idx]
        face_line = faces_lines[face_idx]
        correspondence_line = f"{parts1[0]} {parts2[1]}"

        if correspondence_line not in correct_correspondence:
            correct_correspondence.add(correspondence_line)
            correct_voices.append(voice_line)
            correct_faces.append(face_line)

    print(f"Number of final entries generated: {len(correct_correspondence)}")

    if len(correct_correspondence) < count:
        print(
            f"Warning: Number of generated entries is less than expected. Expected {count}, got {len(correct_correspondence)}.")

    output_file = f'{output_prefix}_train_final_{factor}.txt'
    voices_output_file = f'{output_prefix}_voices_train_filtered_{factor}.csv'
    faces_output_file = f'{output_prefix}_faces_train_filtered_{factor}.csv'

    with open(output_file, 'w') as file:
        for line in correct_correspondence:
            file.write(line + '\n')

    with open(voices_output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(correct_voices)

    with open(faces_output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(correct_faces)

    print(f"Output files saved to: {output_file}")
    print(f"Voice files saved to: {voices_output_file}")
    print(f"Face files saved to: {faces_output_file}")


def main():
    parser = argparse.ArgumentParser(description='Process Urdu training data files.')
    parser.add_argument('--input_prefix', type=str, required=True, help='Prefix for the input files')
    parser.add_argument('--output_prefix', type=str, required=True, help='Prefix for the output files')
    parser.add_argument('--factor', type=float, required=True,
                        help='Factor to determine the number of entries to generate')

    args = parser.parse_args()

    process_files(args.input_prefix, args.output_prefix, args.factor)


if __name__ == '__main__':
    main()
