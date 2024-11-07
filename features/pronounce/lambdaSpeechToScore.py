import torch
import json
import os
# import WordMatching as wm
import utilsFileIO
# import pronunciationTrainer
import base64
import time
# import audioread
import numpy as np
from torchaudio.transforms import Resample
import model
from transformers import Wav2Vec2Processor
from model import CustomWav2Vec2ForCTC  # Import mô hình từ file model.py

checkpoint_dir = r"D:\DOANTOTNGHIEP\code\archive\checkpoint-50000"
processor = Wav2Vec2Processor.from_pretrained(r'D:\DOANTOTNGHIEP\code\archive')
model = CustomWav2Vec2ForCTC.from_pretrained(checkpoint_dir)
import WordMatching
import eng_to_ipa as ipa
import librosa
# trainer_SST_lambda = {}
# trainer_SST_lambda['de'] = pronunciationTrainer.getTrainer("de")
# trainer_SST_lambda['en'] = pronunciationTrainer.getTrainer("en")


# trainer_SST_lambda = {}

def lambda_handler(event, context):

    data = json.loads(event['body'])
    real_text = data['title']
    file_bytes = base64.b64decode(
        data['base64Audio'][22:].encode('utf-8'))
    language = data['language']

    if len(real_text) == 0:
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Credentials': "true",
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': ''
        }

    start = time.time()
    random_file_name = './'+utilsFileIO.generateRandomString()+'.ogg'
    f = open(random_file_name, 'wb')
    f.write(file_bytes)
    f.close()
    print('Time for saving binary in file: ', str(time.time()-start))
    audio_input, sample_rate = librosa.load('000060029.WAV', sr=16000)
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values

    # Chuyển đầu vào và mô hình lên GPU (nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_values = input_values.to(device)
    model.to(device)

    # Thực hiện dự đoán với mô hình
    with torch.no_grad():
        logits = model(input_values, return_dict=True).logits

    # Sử dụng argmax để lấy chỉ số của từ có xác suất cao nhất
    predicted_ids = torch.argmax(logits, axis=-1)

    # Sử dụng processor để giải mã chỉ số thành văn bản
    transcription = processor.batch_decode(predicted_ids)
    print(transcription)

    # # Chuyển đổi danh sách thành chuỗi
    transcription_str = ''.join(transcription)

    # # Loại bỏ các ký tự '[PAD]' và 'h#'
    cleaned_transcription = transcription_str.replace('[PAD]', '')

    print(cleaned_transcription)

    #Matching 
    results = matchSampleAndRecordedWords(real_text,cleaned_transcription )
    # ipa_text = ipa.convert('SIX ONE SIX ZERO')
    # ipa_text = ipa_text.replace("ˈ", "")
    # print(ipa_text)
    # print("Transcription:", cleaned_transcription)
    print(results)
    os.remove(random_file_name)


def matchSampleAndRecordedWords(real_text, ipa_predict):
        
        #convert realtext to ipa
        ipa_text = ipa.convert(real_text)
        ipa_text = ipa_text.replace("ˈ", "")

        real_ipa = ipa_text
        print("ipa affter convert : ", ipa_text)

        model_predict = ipa_predict
        ipa_text = ipa_text.split()
        ipa_predict = ipa_predict.split()


        mapped_words, mapped_words_indices = WordMatching.get_best_mapped_words(
            ipa_predict, ipa_text)


        print(mapped_words)
        print(mapped_words_indices)
        
        is_letter_correct_all_words = ''

        for idx, word_real in enumerate(ipa_text):
            # print(idx)
            # print(word_real)
            mapped_letters, mapped_letters_indices = WordMatching.get_best_mapped_words(
                mapped_words[idx], word_real)

            # print("-------------------------------------------------------------")
            print(mapped_letters)
            # print(mapped_letters_indices)
            
            is_letter_correct = WordMatching.getWhichLettersWereTranscribedCorrectly(
                word_real, mapped_letters)  # , mapped_letters_indices)

            is_letter_correct_all_words += ''.join([str(is_correct)
                                                    for is_correct in is_letter_correct]) + ' '
        
        print(is_letter_correct_all_words)
        return {'ipa_transcript' : real_ipa,
                'ipa predict' : model_predict,
                'matching ipa': is_letter_correct_all_words }


