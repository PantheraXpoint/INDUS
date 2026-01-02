import glob
import os
import json

correct = 0
total = 0
overlap_list = []
time_list = []
postfix = "_full_full_nodes"
for dataset in ["citytour1", "citytour2", "ego1", "ego2", "traffic1", "traffic2", "wildlife1", "wildlife2"]:
    for question_folder in glob.glob(f"ava100_results_M4/{dataset}/q*"):
        if os.path.isdir(question_folder):
            if not os.path.exists(os.path.join(question_folder, f"final_answer{postfix}.json")):
                print(f"No final_answer{postfix}.json in {question_folder}")
                continue
            with open(os.path.join(question_folder, f"final_answer{postfix}.json"), "r") as f:
                data = json.load(f)
                ans = data.get("answer","")
                overlap = data.get("overlap",0)
                if overlap is not None:
                    overlap_list.append(overlap)
                time_list.append(data.get("time_taken",0))
                print(f"Time taken: {data.get('time_taken',0)} seconds")
            with open(f"datas/AVA100/{dataset[:-1]}.json", "r") as f:
                data = json.load(f)
                data = data[int(dataset[-1])-1]
                qa = data.get("qa",[])
                for qa_item in qa:
                    if qa_item.get("question_id") == int(question_folder[-1]):
                        ground_truth = qa_item.get("answer","")
                        break
            if ans == ground_truth:
                correct += 1
            if ans in ["A", "B", "C", "D"]:
                total += 1
            else:
                print(f"Question folder: {question_folder}, Answer: {ans}")
print(f"Correct: {correct}, Total: {total}, Accuracy: {correct/total}, Average Overlap: {sum(overlap_list)/len(overlap_list)}, Average Time: {sum(time_list)/len(time_list)}")
