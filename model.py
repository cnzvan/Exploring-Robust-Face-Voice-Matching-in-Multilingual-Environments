import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion_att(nn.Module):
    def __init__(self, face_input, voice_input, embed_dim_in, mid_att_dim, emb_dim_out):
        super(GatedFusion_att, self).__init__()
        self.linear_face = nn.Sequential()
        self.linear_voice = nn.Sequential()
        self.final_transform = nn.Sequential(
        )
        self.attention = nn.Sequential(
            Forward_Block(embed_dim_in*2, mid_att_dim),
            nn.Linear(mid_att_dim, emb_dim_out)
            )

    def forward(self, face_input, voice_input):
        concat = torch.cat((face_input, voice_input), dim=1)
        attention_out = torch.sigmoid(self.attention(concat))
        face_trans = torch.tanh(self.linear_face(face_input))
        voice_trans = torch.tanh(self.linear_voice(voice_input))
        
        out = face_trans * attention_out + (1.0 - attention_out) * voice_trans
        out = self.final_transform(out)
        
        return out, face_trans, voice_trans
    
class GatedFusion_conv(nn.Module):
    def __init__(self, face_input, voice_input, embed_dim_in, mid_att_dim, emb_dim_out):
        super(GatedFusion_conv, self).__init__()
        self.linear_face = nn.Sequential()
        self.linear_voice = nn.Sequential()
        self.final_transform = nn.Sequential(
        )
        self.attention = nn.Sequential(
            Forward_Block_conv(embed_dim_in*2, mid_att_dim),
            nn.Linear(mid_att_dim, emb_dim_out)
            )

    def forward(self, face_input, voice_input):
        concat = torch.cat((face_input, voice_input), dim=1)
        attention_out = torch.sigmoid(self.attention(concat))
        face_trans = torch.tanh(self.linear_face(face_input))
        voice_trans = torch.tanh(self.linear_voice(voice_input))
        
        out = face_trans * attention_out + (1.0 - attention_out) * voice_trans
        out = self.final_transform(out)
        
        return out, face_trans, voice_trans

class Forward_Block_att(nn.Module):
    
    def __init__(self, input_dim=128, output_dim=128, p_val=0.0):
        super(Forward_Block_att, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=p_val)
        )
    def forward(self, x):
        return self.block(x)


class Forward_Block_conv(nn.Module):
    def __init__(self, input_dim=128, output_dim=128, p_val=0.0):
        super(Forward_Block_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=p_val)
        )
        self.attention = nn.Sequential(
            nn.Conv1d(output_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(2)  
        x = self.conv(x)
        attention_weights = self.attention(x)
        x = x * attention_weights
        x = x.squeeze(2)  
        return x
    
def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out), 
                        nn.BatchNorm1d(f_out),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5))


class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim).cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.normalize(x)
        return x



class FOP_frozen(nn.Module):
    def __init__(self, args, face_feat_dim, voice_feat_dim, n_class):
        super(FOP_frozen, self).__init__()
        
        self.voice_branch = EmbedBranch(voice_feat_dim, args.dim_embed)
        self.face_branch = EmbedBranch(face_feat_dim, args.dim_embed)
        
        if args.fusion == 'linear':
            self.fusion_layer = LinearWeightedAvg(args.dim_embed, args.dim_embed)
        elif args.fusion == 'gated':
            self.fusion_layer = GatedFusion_att(face_feat_dim, voice_feat_dim, args.dim_embed, 128, args.dim_embed)
        
        self.logits_layer = nn.Linear(args.dim_embed, n_class)

        if args.cuda:
            self.cuda()

    def forward(self, faces, voices):
        voices = self.voice_branch(voices)
        faces = self.face_branch(faces)
        feats, faces, voices = self.fusion_layer(faces, voices)
        logits = self.logits_layer(feats)
        
        return [feats, logits], faces, voices
    
    def train_forward(self, faces, voices, labels):
        
        comb, face_embeds, voice_embeds = self(faces, voices)
        return comb, face_embeds, voice_embeds

class FOP_update(nn.Module):
    def __init__(self, args, face_feat_dim, voice_feat_dim, n_class):
        super(FOP_update, self).__init__()
        
        self.voice_branch = EmbedBranch(voice_feat_dim, args.dim_embed)
        self.face_branch = EmbedBranch(face_feat_dim, args.dim_embed)
        
        if args.fusion == 'linear':
            self.fusion_layer = LinearWeightedAvg(args.dim_embed, args.dim_embed)
        elif args.fusion == 'gated':
            self.fusion_layer = GatedFusion_conv(face_feat_dim, voice_feat_dim, args.dim_embed, 128, args.dim_embed)
        
        self.logits_layer = nn.Linear(args.dim_embed, n_class)

        if args.cuda:
            self.cuda()

    def forward(self, faces, voices):
        voices = self.voice_branch(voices)
        faces = self.face_branch(faces)
        feats, faces, voices = self.fusion_layer(faces, voices)
        logits = self.logits_layer(feats)
        
        return [feats, logits], faces, voices
    
    def train_forward(self, faces, voices, labels):
        
        comb, face_embeds, voice_embeds = self(faces, voices)
        return comb, face_embeds, voice_embeds


'''
 W
'''
class CombinedFOPW(nn.Module):
    def __init__(self, args, fop_frozen, face_feat_dim2, voice_feat_dim2, n_class):
        super(CombinedFOPW, self).__init__()
        self.fop_frozen = fop_frozen
        for param in self.fop_frozen.parameters():
            param.requires_grad = False

        self.fop_update = FOP_update(args, face_feat_dim2, voice_feat_dim2, n_class)
        
        self.w = nn.Parameter(torch.ones(1, args.dim_embed))
        self.logits_layer = nn.Linear(args.dim_embed, n_class)
        
        if args.cuda:
            self.cuda()

    def forward(self, face_input, voice_input):

        comb1, face_embeds1, voice_embeds1 = self.fop_frozen(face_input, voice_input)
        feats1, _ = comb1

        comb2, face_embeds2, voice_embeds2 = self.fop_update(face_input, voice_input)
        feats2, logits2 = comb2

        W = torch.sigmoid(self.w) 
        feats = W * feats1 + (1 - W) * feats2

        logits = self.logits_layer(feats)
        
        return [feats, logits], face_embeds2, voice_embeds2

    def train_forward(self, face_input, voice_input, labels):
        comb, face_embeds, voice_embeds = self.forward(face_input, voice_input)
        return comb, face_embeds, voice_embeds
'''
Attention
'''
class CombinedFOPAtt(nn.Module):
    def __init__(self, args, fop_frozen, face_feat_dim2, voice_feat_dim2, n_class):
        super(CombinedFOPAtt, self).__init__()
        self.fop_frozen = fop_frozen
        for param in self.fop_frozen.parameters():
            param.requires_grad = False

        self.fop_update = FOP_update(args, face_feat_dim2, voice_feat_dim2, n_class)
        
        self.attention = nn.Sequential(
            Forward_Block(args.dim_embed * 2, args.dim_embed),
            nn.Linear(args.dim_embed, 1),
            nn.Sigmoid()
        )
        self.logits_layer = nn.Linear(args.dim_embed, n_class)
        
        if args.cuda:
            self.cuda()

    def forward(self, face_input, voice_input):
        comb1, face_embeds1, voice_embeds1 = self.fop_frozen(face_input, voice_input)
        feats1, _ = comb1

        comb2, face_embeds2, voice_embeds2 = self.fop_update(face_input, voice_input)
        feats2, logits2 = comb2

        concat_feats = torch.cat((feats1, feats2), dim=1)
        attention_weights = self.attention(concat_feats)
        
        feats = attention_weights * feats1 + (1 - attention_weights) * feats2
        logits = self.logits_layer(feats)
        
        return [feats, logits], face_embeds2, voice_embeds2

    def train_forward(self, face_input, voice_input, labels):
        comb, face_embeds, voice_embeds = self.forward(face_input, voice_input)
        return comb, face_embeds, voice_embeds

'''
Conv 
'''
class CombinedFOPConv(nn.Module):
    def __init__(self, args, fop_frozen, face_feat_dim2, voice_feat_dim2, n_class):
        super(CombinedFOPConv, self).__init__()
        self.fop_frozen = fop_frozen
        for param in self.fop_frozen.parameters():
            param.requires_grad = False

        self.fop_update = FOP_update(args, face_feat_dim2, voice_feat_dim2, n_class)
        
        self.feature_fusion_block = FeatureFusionBlock(input_dim=args.dim_embed, output_dim=args.dim_embed, p_val=0.0)
        self.logits_layer = nn.Linear(args.dim_embed, n_class)
        
        if args.cuda:
            self.cuda()

    def forward(self, face_input, voice_input):

        comb1, face_embeds1, voice_embeds1 = self.fop_frozen(face_input, voice_input)
        feats1, _ = comb1

        comb2, face_embeds2, voice_embeds2 = self.fop_update(face_input, voice_input)
        feats2, logits2 = comb2

        feats = torch.stack([feats1, feats2], dim=1)  
        feats = feats.permute(0, 2, 1)  
        fused_feats = self.feature_fusion_block(feats)

        logits = self.logits_layer(fused_feats)
        
        return [fused_feats, logits], face_embeds2, voice_embeds2

    def train_forward(self, face_input, voice_input, labels):
        comb, face_embeds, voice_embeds = self.forward(face_input, voice_input)
        return comb, face_embeds, voice_embeds
