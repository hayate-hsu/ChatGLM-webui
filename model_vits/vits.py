# coding=utf-8
from scipy.io.wavfile import write
import time
import os
import gradio as gr
from model_vits import utils
from model_vits import commons
from model_vits.models import SynthesizerTrn
from model_vits.text import text_to_sequence
import torch
from torch import no_grad, LongTensor
import logging
import datetime

class Vits:
    device = 'cpu'
    hps_ms = None
    net_g_ms = None
    model = None 
    optimizer= None 
    learning_rate = None 
    epochs = None
    ns = 0.6
    nsw = 0.688
    ls = 1.2
    speakers = 0
    spk_id = 226
    voice_path = './outputs/voices'
    def __init__(self,voice_path='./outputs/voices') -> None:
        logging.getLogger('numba').setLevel(logging.ERROR)
        # 
        self.voice_path = voice_path
        if not os.path.exists(self.voice_path):
            os.makedirs(self.voice_path)
            
        limitation = os.getenv("SYSTEM") == "spaces"  # limit text and audio length in huggingface spaces
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hps_ms = utils.get_hparams_from_file(r'./model_vits/config.json')
        self.net_g_ms = SynthesizerTrn(
            len(self.hps_ms.symbols),
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.hps_ms.data.n_speakers,
            **self.hps_ms.model)
        _ = self.net_g_ms.eval().to(self.device)
        self.speakers = self.hps_ms.speakers
        self.model, self.optimizer, self.learning_rate, self.epochs = utils.load_checkpoint(r'./model_vits/G_953000.pth', self.net_g_ms.to(self.device), None)
    
    def get_spks(self):
        return self.speakers
    
    def set_spk(self,spk):
        self.spk_id = self.speakers.index(spk)
        
    def get_text(self, text, hps):
        text_norm, clean_text = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm, clean_text

    def vits(self, text, language, speaker_id, noise_scale, noise_scale_w, length_scale, device, hps_ms, net_g_ms):
        start = time.perf_counter()
        if not len(text):
            return "输入文本不能为空", None, None
        text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
        if len(text) > 500 and limitation:
            return f"输入文字过长！{len(text)}>500", None, None
        if language == 0:
            text = f"[ZH]{text}[ZH]"
        elif language == 1:
            text = f"[JA]{text}[JA]"
        else:
            text = f"{text}"
        stn_tst, clean_text = self.get_text(text, hps_ms)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            speaker_id = LongTensor([speaker_id]).to(device)
            #audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=speaker_id, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
            #                    length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=speaker_id, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                length_scale=length_scale)[0][0, 0].data.float().detach().cpu().numpy()

        return audio

    def generateSound(self, text, store_path='', ns=0.6, nsw=0.688, ls=1.2, ):
        if text:
            # result = self.vits(text, 0, 226, self.ns, self.nsw, self.ls, self.device, self.hps_ms, self.net_g_ms)
            result = self.vits(text, 0, self.spk_id, self.ns, self.nsw, self.ls, self.device, self.hps_ms, self.net_g_ms)
            date = datetime.datetime.now() 
            file_name = date.strftime('%Y-%m-%d-%H-%M-%S')
            store_path = self.voice_path + '/' + file_name+''
            write(store_path, 22050, result)
            return result, store_path
        
        return result,''
if __name__ == '__main__':
    vits = Vits()
    vits.generateSound('你好，我是爱莉希雅，很高兴见到你', './demo2.wav')
