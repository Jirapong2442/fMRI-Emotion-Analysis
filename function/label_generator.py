
import numpy as np
import pandas as pd
import re
from transformers import AutoModelForSequenceClassification
import os
from transformers import AutoTokenizer, AutoConfig, AutoModel
from scipy.special import softmax

#load model
MODEL = f"j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(MODEL)

def preprocess_subtitle(path):
    with open(path, "r", encoding='utf-8-sig') as f:
        string = f.read()
        lines = string.split("\n")
        time_arr = []
        subtitles = []
        first_text = True # to indicate the first text after time
        continueable_dot = False # for "..." at the end of line to indicate continuation
        #continueable_line = False # for line that is continued in next line without "..."
        for line in lines:
            if (len(line) < 4 and line.isdigit()) or len(line) == 0:
                continue
            
            
            if " --> " not in line:
                text = line.strip()
                if continueable_dot:
                    last_subtitles = subtitles[-1]
                    subtitles.pop()
                    subtitles.append((last_subtitles + " " + text))

                    if  first_text: #  if ... doesnt have anything after it
                        last_time = time_arr[-2]
                        time_arr.pop()
                        time_arr.pop()
                        time_arr.append((last_time[0], end_time))
                    continueable_dot = False
                    first_text = False

                #elif continueable_line:
                #    last_subtitles = subtitles[-1]
                #    subtitles.pop()
                #    subtitles.append((last_subtitles + " " + text))
                #    continueable_line = False
                #    first_text = False

                else:
                    if "..." in line:
                        text = text.replace("...", "")
                        continueable_dot = True
                        #subtitles.append(text)

                    if first_text: #and continueable_dot == False:
                        subtitles.append(text)
                        first_text = False
                        #continueable_line = True
                    else:
                        last_subtitles = subtitles[-1]
                        subtitles.pop()
                        subtitles.append((last_subtitles + " " + text))

                if len(subtitles) != len(time_arr):
                    print("Warning: subtitle and time length mismatch" f"({len(subtitles)} vs {len(time_arr)}) at line: {line}")
                    print("Subtitle so far:", subtitles)
                    print("Time so far:", time_arr)

                

            else:
                first_text = True
                continueable_line = False
                time = line.split(" --> ")
                start = time[0].split(",")[0].split(":")
                end = time[1].split(",")[0].split(":")
                start_ms = time[0].split(",")[1]
                end_ms = time[1].split(",")[1]
                start_time = int(start[0]) * 3600 + int(start[1]) * 60 + float(start[2]) + float(start_ms) / 1000
                end_time = int(end[0]) * 3600 + int(end[1]) * 60 + float(end[2]) + float(end_ms) / 1000
                time_arr.append((start_time, end_time))

            

    sentence_arr = pd.DataFrame({
    "start_time": [t[0] for t in time_arr],
    "end_time": [t[1] for t in time_arr],
    "text": subtitles
    })

    return time_arr, subtitles, sentence_arr # first two for debug, last for main use

def clean_sentence(sentence):
    sentence = re.sub(r'\W+', ' ', sentence)  # Remove symbols/punctuation
    sentence = sentence.lower()               # Lowercase
    sentence = re.sub(r'\s+', ' ', sentence)  # Remove extra spaces
    return sentence.strip()  

def get_emotion_scores(sentence):
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model_sentiment(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores

def run_emotion_label_generator(subtitle_path, output_path=None):
    file_name = subtitle_path.split("/")[-1].replace("_subtitle.srt", "")
    out_path = output_path if output_path is not None else "./"
    file_name = os.path.join(out_path, file_name + "_emotion.csv")

    _,_,sentence_arr = preprocess_subtitle(subtitle_path)
    sentence_arr['text'] = sentence_arr['text'].apply(clean_sentence)
    emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    emotion_scores = sentence_arr['text'].apply(get_emotion_scores)
    emotion_df = pd.DataFrame(emotion_scores.tolist(), columns=emotions)
    result_df = pd.concat([sentence_arr, emotion_df], axis=1)
    result_df.to_csv(file_name, index=False)


if __name__ == "__main__":
    path = "/mnt/c/Users/NA/Saved Games/fmri/subtitle_all/little_miss_sunshine_subtitle.srt"
    out_path = "/home/jirapong/fmri_emotion/fMRI-Emotion-Analysis/emotion_subtitle/"
    run_emotion_label_generator(path, out_path)