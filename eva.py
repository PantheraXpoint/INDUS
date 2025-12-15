import glob
import os
import json

correct = 0
total = 0
for dataset in ["citytour1", "citytour2", "ego1", "ego2", "traffic1", "traffic2", "wildlife1", "wildlife2"]:
    for question_folder in glob.glob(f"ava100_results/{dataset}/q*"):
        if os.path.isdir(question_folder):
            if not os.path.exists(os.path.join(question_folder, "final_answer.json")):
                print(f"No final_answer.json in {question_folder}")
                continue
            with open(os.path.join(question_folder, "final_answer.json"), "r") as f:
                data = json.load(f)
                ans = data.get("answer","")
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
            total += 1
print(f"Correct: {correct}, Total: {total}, Accuracy: {correct/total}")