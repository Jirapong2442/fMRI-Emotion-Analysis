
import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    path = "/mnt/c/Users/NA/Saved Games/eeg_study/subtitle_all/little_miss_sunshine_subtitle.srt"
    _,_,sentence_arr = preprocess_subtitle(path)
    print(sentence_arr)
    