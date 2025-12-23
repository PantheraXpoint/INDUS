import glob
import os
import json

correct = 0
total = 0
overlap_list = []
time_list = []
for dataset in ["citytour1", "citytour2", "ego1", "ego2", "traffic1", "traffic2", "wildlife1", "wildlife2"]:
    for question_folder in glob.glob(f"ava100_results/{dataset}/q*"):
        if os.path.isdir(question_folder):
            if not os.path.exists(os.path.join(question_folder, "final_answer.json")):
                print(f"No final_answer.json in {question_folder}")
                continue
            with open(os.path.join(question_folder, "final_answer.json"), "r") as f:
                data = json.load(f)
                ans = data.get("answer","")
                overlap = data.get("overlap",0)
                if overlap is not None:
                    overlap_list.append(overlap)
                time_list.append(data.get("time_taken",0))
            with open(f"datas/AVA100/{dataset[:-1]}.json", "r") as f:
                data = json.load(f)
                data = data[int(dataset[-1])-1]
                qa = data.get("qa",[])
                for qa_item in qa:
                    if qa_item.get("question_id") == int(question_folder[-1]):
                        ground_truth = qa_item.get("answer","")
                        break
            if ans == ground_truth and overlap is not None and overlap > 0:
                correct += 1
            if ans in ["A", "B", "C", "D"] and overlap is not None and overlap > 0:
                total += 1
print(f"Correct: {correct}, Total: {total}, Accuracy: {correct/total}, Average Overlap: {sum(overlap_list)/len(overlap_list)}, Average Time: {sum(time_list)/len(time_list)}")