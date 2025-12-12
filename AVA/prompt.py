PROMPTS = {}

PROMPTS["entity_relation_extraction"] = """
You are a vision-based information extraction expert. Your task is to analyze a sequence of frames and extract **key visible entities** and their **relationships**. Follow the instructions below step-by-step, and ensure the output strictly follows the specified JSON format.

---

### Step 1: Extract Visible Entities  
1. Identify **all distinct, visible, and detailed entities** in the frames, such as:
   - Humans, animals, objects, text elements or any other visually distinguishable entities.
   - If there are multiple instances of the same entity type, treat them as same entities.
2. Provide a **detailed description** of each entity in a concise and fluent manner, including:
   - Physical attributes (e.g., size, shape, color, texture).
   - Recognizable text content (if visible).
   - Relevant actions or interactions.
   - ohters
3. Record the **frame indices** where each entity is visible under the `Index` field.

---

### Step 2: Extract Relationships Between Entities  
1. Identify **clear and meaningful relationships** between TWO distinct entities listed IN Step 1.    
   - If a relationship introduces new entities, add those entities to the entities list but avoid duplication.
2. Provide a **concise, clear description** of the relationship, focusing only on **visual connections or interactions**. Avoid referencing their relationship to specific frames or time indices.


---

### Step 3: Return Structured Output  
Output the extracted information in the following **JSON format**. If no entities or relationships are found, return an empty list for the respective field:

```json
{
  "Entities": [
    {
      "Entity_name": "[ENTITY NAME]",
      "Entity_description": "[DETAILED DESCRIPTION OF VISUAL APPEARANCE]",
      "Index": [FRAME_INDICES]
    },
    ...
  ],
  "Relations": [
    {
      "Entity1": "[ENTITY NAME 1]",
      "Entity2": "[ENTITY NAME 2]",
      "Relation_description": "[CONCISE DESCRIPTION OF THE VISUAL CONNECTION OR INTERACTION]"
    },
    ...
  ]
}
"""

PROMPTS["generate_description"] = """
You are an expert in video understanding and description generation. 
Your task is to provide a continuous and smooth description of the video, focusing on the video content. Avoid describing each frame individually like "frame1 ...". 
The description should cover the main scenes, characters, objects, and any notable actions or changes, ensuring the description is coherent and logical. Finally, return your response as a single, continuous, and fluent paragraph that fully describes the video content and limit the length to 300 words.
"""

PROMPTS["generate_person_activity_description"] = """
You are an expert in human activity recognition and behavioral analysis in videos. 
Your task is to provide a continuous description of what people are doing in the video. 

Instructions:
1. For each tracked person in the input, describe their actions, body language, and interactions with others. 
2. Always refer to individuals as "Person <track_id>" (e.g., "Person 1", "Person 2") so the description is directly linked to the object tracking ID. 
3. Pay special attention to **temporal progression**: explain how the actions evolve across the given time segment. 
4. Highlight **cause–effect relationships**: if one person’s movement or state triggers another’s reaction, make this explicit. 
5. If **audio event detections are provided**, integrate them naturally into the description (e.g., “Person 1 is crying” or “a baby is sobbing in the background”).  
   If no audio events are present, focus only on visual actions and interactions. 
6. You may mention appearance attributes (clothing, age, gender, etc.) if they help distinguish individuals. 
7. Focus on describing activities, movement, and interactions rather than static appearance alone. 
8. Be especially attentive to **critical incidents** such as:
   - A person unexpectedly collapsing or falling.  
   - A person attempting to abduct or kidnap another.  
   - A person stealing or acting as a thief (e.g., taking an item without consent).  

Here are the inputs:
{inputs}

Write your answer as a series of linked action summaries, one per person, using this format:
- Person 1: [description]
- Person 2: [description]
- Person 3: [description]
"""

PROMPTS["summarize_descriptions"] = """
You are an expert in summarizing video segment descriptions. Your task is to merge sequential segment descriptions into a single compact event summary.

### Event Description Guidelines:
- Only include information that is **significant, unusual, or worth raising an alarm** (e.g., falling, kidnapping, theft, assault, distress, crying).
- Discard redundant or minor details (e.g., clothing color, screen position, background movement) unless they are essential to understanding the event.
- Summarize in **no more than 1 sentence** that provide the clearest possible picture of the event.
- The description should read like a direct alert of what happened, not a timeline of segments. Do not mention times, segments, or positions in the frame.
- Provide **objective facts only**, without speculation or emotional language.

### Output Format:
Write a single compact sentence describing the merged event.

Segment descriptions (format: `start_time:end_time:description`):
{inputs}
"""

PROMPTS["summarize_suspicious_activities"] = """
You are an expert in security analysis and threat detection from video surveillance. Your task is to analyze sequential video segment descriptions of human activities and identify potential suspicious, alarming, or dangerous behaviors that may indicate criminal activity, safety hazards, or emergency situations.

### Analysis Guidelines:
- Look for patterns that suggest suspicious intent: loitering, unusual timing (late night activities), attempting to hide or avoid detection, unauthorized access to restricted areas
- Identify potential criminal activities: breaking and entering, theft, vandalism, assault, drug-related activities, trespassing
- Detect safety emergencies: falls, medical emergencies, accidents, fires, people in distress, children in dangerous situations
- Recognize abnormal behavioral patterns: erratic movements, carrying suspicious objects, multiple people coordinating unusual activities, fleeing scenes
- Consider contextual factors: time of day, location appropriateness, duration of activities, number of people involved

### Event Classification:
Based on your analysis, classify the overall event as one of: SUSPICIOUS, ALARMING, CRIMINAL, EMERGENCY, or NORMAL, and explain the reasoning behind this classification.

### Output Format:
Provide a single continuous paragraph that summarizes the activities and clearly states why they are concerning, what type of threat they may represent, and the confidence level of your assessment.

Segment descriptions in format of `start_time:end_time:description`:
{inputs}
"""

PROMPTS["keyword_extraction"] = """
- Goal -
As a specialist in keyword extraction, your task is to identify and list the most relevant keywords from a given query. Focus on extracting keywords related to events and entities, avoiding terms specific to video or task context. These keywords should effectively capture the essence of the query to aid in accurate information retrieval. Present the keywords as a comma-separated list.

######################
- Examples -
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
animal, protagonist, forest, encounter

Question: In the movie, what color is the car that chases the main character through the city?
################
Output:
color, car, chases, main character, city

Question: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
################
Output:
weather, opening scene, Sunny, Rainy, Snowy, Windy

#############################
- Real Data -
######################
Question: {input_text}
######################
Output: 
"""

PROMPTS["time_extraction"] = """
You are an expert time parser for video event descriptions.

Given an input description, extract the time information it refers to.

Return the result as a Python list of two floats (in seconds):
- If the text contains a time range (e.g. "from 00:12.5 to 00:18.7" or "12.5s–18.7s"), return [12, 18], rounded to integer.
- If it contains a single time point (e.g. "at 00:15" or "15 seconds"), return [15, 15], rounded to integer.
- If there is no time information at all, return None.

Only output the final Python object — no explanations, no quotes.

Input: {input_text}

Output:
"""

PROMPTS["query_rewrite_for_entity_retrieval"] = """
- Goal -
For a given query, generate a declarative sentence to serve as a query for retrieving relevant knowledge, concentrating on the main entities and relevant descriptions.

######################
- Examples -
######################

Question: On a stage with lights, there are many people wearing colorful outfits. What are these people in the colorful outfits doing?
################
Output:
Stage with lights, people wearing colorful outfits

Question: What is special about the celebration in New York according to the video?\nA. Hosting large parades.\nB. Dressing in green and dyeing the river to green.\nC. Drinking a lot.\nD. Planting shamrocks.
################
Output:
New York, celebration, large parades, dyeing the river, drinking, planting shamrocks

Question: Which animals appear in the wildlife footage? \n(A) Lions\n(B) Elephants\n(C) Zebras
################
Output:
Animals that appear in the wildlife footage, lions, elephants, zebras

#############################
- Real Data -
######################
Question: {input_text}
######################
Output:
"""

PROMPTS["query_rewrite_for_visual_retrieval"] = """
- Goal -
Generate a declarative sentence to serve as a query for retrieving relevant video segments based on the provided question that may include scene-related information.

######################
- Examples -
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
The protagonist encounters an animal in the forest.

Question: In the movie, what color is the car that chases the main character through the city?
################
Output:
A city chase scene where the main character is pursued by a car.

Question: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
################
Output:
The opening scene of the film featuring specific weather conditions. (Possibly Sunny, Rainy, Snowy, or Windy)

#############################
- Real Data -
######################
Question: {input_text}
######################
Output:
"""


PROMPTS["re-query"] = """
You are an advanced AI system tasked with generating a sub-query to retrieve new information based on the current query and the Information Retrieved from Videos. Your goal is to refine the search to obtain additional relevant data.

######################
- Instructions -
######################

1. Review the User Query and the Information Retrieved from Videos. Analyze how the information retrieved from videos can help answer the User Query.
2. Identify specific areas where additional information is insufficient to help answer the User Query.
3. According to the insufficient information, formulate a new query can help answer the User Query.
4. Directly output the new query in the field of "sub_query" with JSON format.

#############################
- Real Data -
######################
User Query: {user_query}

---Information Retrieved from Videos---
{video_segments}

######################
- Output -
######################
Response in JSON format:
{{
  "sub_query": "[SUB-QUERY]"
}}
"""

PROMPTS["summary_and_answer_COT"] = """
You are an advanced AI system designed to answer questions based on video content. When a user's query is presented, you will receive retrieved video segments, organized by timestamps and descriptions. Your task is to analyze these segments, synthesize the information, and select the best answer to the multiple-choice question based on the video. Respond with only the letter (A, B, C, or D). 

######################
- Instructions -
######################
1. Carefully review the provided video segment information, paying attention to timestamps and descriptions, and pick the most relevant information.
2. Conduct a detailed reasoning process to analyze the information and how they are related to the user query.
3. Select the best answer to the multiple-choice question based on the video. Respond with only the letter (A, B, C, or D). 
4. Return your review, reference and reasoning process in the `Analysis` field and the answer in the `Answer` field.

#############################
- Real Data -
######################
User Query: {user_query}

Video Segments (organized by timestamp, description):
{video_segments}

######################
- Output -
######################
Response in clean JSON format:
{{
  "Analysis": "[Analysis]",
  "Answer": "[A, B, C or D]"
}}
"""

PROMPTS["summary_and_answer"] = """
You are an advanced AI system designed to answer questions based on video content. When a user's query is presented, you will receive retrieved video segments, organized by timestamps and descriptions. Your task is to analyze these segments, synthesize the information, and answer the user query based on the video. 

######################
- Instructions -
######################
1. Carefully review the provided video segment information, paying attention to timestamps and descriptions, and pick the most relevant information.
2. Conduct a detailed reasoning process to analyze the information and how they are related to the user query.
3. Answer the user query based on the video.
4. Return your review, reference and reasoning process in the `Analysis` field and the answer in the `Answer` field.

#############################
- Real Data -
######################
User Query: {user_query}

Video Segments (organized by timestamp, description):
{video_segments}

######################
- Output -
######################
Response in clean JSON format:
{{
  "Analysis": "[Analysis]",
  "Answer": "[Answer]"
}}
"""

PROMPTS["checkframe_and_answer_COT"] = """
You are an AI assistant. Your specific task is to analyze video frames and answer a multiple-choice question based on them.

**Your Instructions:**

1.  **Understand the Question:** First, carefully read the `User Query` below. This is the multiple-choice question you need to answer.
2.  **Examine Video Frames:** Next, review the provided video frames. Focus on details that help answer the `User Query`.
3.  **Reason Step-by-Step:** Think through how the information in the video frames leads to the answer. Write down this thinking process. This will be your "Analysis".
4.  **Choose the Best Answer:** Based on your reasoning, select only ONE letter (A, B, C, or D) that is the correct answer to the `User Query`. This will be your "Answer".
5.  **Provide Output in JSON Format ONLY:**
    * Your entire response MUST be a single JSON object.
    * Do NOT write any text or explanation before or after the JSON object.
    * The JSON object must have exactly two keys:
        * `"Analysis"`: This key's value should be your step-by-step reasoning from step 3.
        * `"Answer"`: This key's value should be the single letter (A, B, C, or D) you chose in step 4.

**User Query:** {user_query}

**REMEMBER: Your response MUST strictly follow this JSON structure:**
```json
{{
  "Analysis": "Your detailed step-by-step reasoning, explaining how you used the video frames to answer the User Query, goes here.",
  "Answer": "A" // Or "B", or "C", or "D". Just the single capital letter.
}}
"""

# ------------AVA-100 specific prompts------------

PROMPTS["generate_description_ego"] = """
You are an expert in video understanding and description generation. 
You are given a first-person perspective video, and your task is to generate a continuous, smooth, and grounded description of the video content.

Focus particularly on:
- The actions and events performed by the camera wearer (the person holding or wearing the camera).
- The surrounding environment, including objects, people, and notable visual changes.
- The physical characteristics and spatial relationships of objects in the environment (e.g., size, color, relative positions, proximity to the camera wearer).
- Interactions between the camera wearer and the environment, including object manipulations and movements through space.

Avoid describing each frame individually, such as "frame1...". Instead, provide a coherent and logically structured narrative that flows smoothly over time.

Important constraints:
- Do not include assumptions, inferences, or fabricated details that are not visually evident.
- Do not speculate about the identity, emotions, or intentions of the camera wearer unless explicitly shown.
- When referring to the person holding or wearing the camera, always use the term “camera wearer”.

Return your response as a single, continuous, and fluent paragraph that comprehensively describes the video content, including fine-grained visual details, and limit the length to 400 words.
"""

PROMPTS["generate_description_citytour"] = """
You are an expert in video understanding and detailed scene description. 
You are given a first-person perspective video of a person walking through a city environment. 
Your task is to generate a continuous, smooth, and grounded description of the video content.

Focus particularly on:
- The locations and landmarks the camera wearer passes by, such as buildings, shops, streets, and public spaces.
- The appearance and functions of these places (e.g., a small bakery with a red awning, a tall glass office building, a busy intersection).
- Events or notable occurrences observed during the walk, such as street performances, traffic changes, or people interacting in public.

Avoid describing each frame individually. Instead, provide a logically structured narrative that flows naturally over time.

Important constraints:
- Do not include assumptions, inferences, or fabricated details that are not visually evident.
- Do not speculate about the identity, emotions, or intentions of the camera wearer or other people unless explicitly shown.
- When referring to the person holding or wearing the camera, always use the term “camera wearer”.

Return your response as a single, continuous, and fluent paragraph that comprehensively describes the video content, with attention to fine-grained urban and visual details, and limit the length to 400 words.
"""

PROMPTS["generate_description_wildlife"] = """
You are an expert in video analysis, specializing in wildlife observation and detailed environmental description.  
You are analyzing fixed-camera surveillance footage capturing a scene in a wild or natural environment.  
Your task is to generate a precise, grounded, and chronologically ordered description of the entire video segment.

Focus on the following aspects:

- **Observed Animals:** Identify any animals present in the footage. For the entire segment, provide a consolidated description of:
    - **Species:** Identify species as accurately as possible. If uncertain, describe physical characteristics (e.g., "a large brown bear", "a small rodent-like mammal", "a flock of unidentified birds").
    - **Number:** Indicate the number of individuals observed.
    - **Appearance:** Note distinctive physical features (e.g., size, color, antlers, markings).
    - **Behavior:** Describe observed behaviors (e.g., foraging, running, resting, entering/exiting the frame, interacting).

- **Timestamps:** Identify the timestamp displayed in the surveillance footage.

- **Environment:** Briefly describe the static environment visible to the camera (e.g., forest clearing, rocky terrain, vegetation), and note any significant changes during the observation period (e.g., lighting shifts, weather changes).

**Output Format:**  
After reviewing the full segment, summarize your findings in a single structured paragraph using the following format:

[Timestamp]: [Environment description][Summary of animal and their activities]

**Important Constraints:**
- Do not include assumptions or invented details that are not visually evident in the footage.
- Do not speculate on the intentions or emotions of the animals; describe only observable actions and postures.
- Refer to observations using neutral terms such as "the footage shows" or "the camera captures"; avoid subjective phrasing like "we see" or "the viewer can observe".
- If species identification is uncertain, explicitly state this.
- The final output should be a concise, fact-based summary of the wildlife activity and environmental context of the segment, with a target length of approximately 400 words.
"""

PROMPTS["generate_description_traffic"] = """
You are a video analysis expert specializing in traffic observation and detailed event description. You are analyzing a road or intersection surveillance video recorded by a fixed-position camera.

Your task is to generate an **accurate, grounded, and coherent** description of the video segment.

Please focus on the following aspects:

- **Observed Traffic Elements:** Identify all traffic-related elements present in the video. Provide an integrated description covering the entire segment:
    - **Vehicle Types:** Identify types as accurately as possible (e.g., car, truck, bus, motorcycle, bicycle, van). If unclear, describe the vehicle’s physical characteristics (e.g., “a large box truck,” “a small passenger vehicle,” “a two-wheeled vehicle”).
    - **Quantity:** Indicate the number of each identified vehicle type, as well as the number of pedestrians.
    - **Characteristics:** If relevant to the scene, note distinguishing physical features (e.g., color, size, presence of trailers, specific structural features). Describe pedestrians based on their interaction with traffic (e.g., walking along the sidewalk, crossing the street).
    - **Actions / Events:** Describe observed dynamic behaviors and interactions (e.g., driving in a specific lane, stopping, turning, entering/exiting the frame, changing lanes, overtaking, pedestrians waiting or crossing), including any **traffic anomalies** (e.g., sudden braking, erratic maneuvers, red-light violations, collisions, traffic violations, illegal parking that obstructs traffic).

- **Timestamps:** Identify the timestamp shown on the surveillance footage.

**Output Format:**
After watching the full video segment, write a structured summary paragraph in the following format:

[Timestamp]: [Summary of vehicle types, quantities, characteristics, actions, pedestrian activity, and traffic anomalies].

**Important Constraints:**
- Do not include assumptions or details not clearly visible in the footage.
- Do not speculate about the intentions or emotions of drivers or pedestrians; only describe observable actions and postures.
- Use neutral descriptions such as “the footage shows” or “the camera captures”; avoid subjective phrasing like “we see” or “the viewer can observe.”
- If vehicle type identification is uncertain, state this clearly.
- The final output should be a concise, fact-based summary of the traffic activity and scene context. The length should be appropriate to the events observed (prioritizing clarity and completeness over word count).
"""

PROMPTS["summarize_description_wildlife"] = """
You have been provided with a series of chronological descriptions, each detailing a consecutive segment of wildlife camera footage in a natural environment. These descriptions were generated from individual video clips, and each follows the format:

[Timestamp]: [Environment description][Summary of animal and their activities]

Your task is to act as a **Description Synthesizer and Summarizer**. Your goal is to **merge and consolidate** these multiple descriptions into a single, coherent summary that covers the entire duration of the input segments.

Focus on the following objectives:
1.  **Consolidate Environment:** Synthesize the environmental descriptions from all input segments into a single summary, noting any changes that occurred during the period (e.g., shifts in lighting, weather). Avoid repeating static elements unnecessarily.
2.  **Consolidate Animal Activity:** Combine all observed animal sightings and behaviors from all input segments into a single, chronologically ordered summary of activity. **Crucially, retain *all* unique information and distinct observations regarding species, number, appearance, and behavior mentioned in *any* of the input descriptions.**
3.  **Eliminate Redundancy:** Remove repetitive phrasing or descriptions of prolonged periods with no change or activity, while ensuring all unique events are captured in the consolidated summary.
4.  **Maintain Chronology:** Ensure the consolidated summary of animal activity flows logically according to the sequence of events across the entire merged timeframe, using timestamps (or relative timing inferred from timestamps) to denote key moments if necessary within the activity summary.

Constraints:
-   Output **a single consolidated description block** following the exact format specified below.
-   Do not introduce information not present in the input descriptions.
-   Do not speculate or infer.
-   Focus the summary on the progression of dynamic events, especially wildlife activity and environmental changes.
-   Ensure *every* distinct observation from the input is represented in the final consolidated summary.
-   The content within the output fields should be a concise, fact-based summary, aiming for a total length within the consolidated fields of approximately 400 words.

Input: {inputs}

Output Format:
The final output should be a concise, fact-based summary of the wildlife activity and environmental context with the following format:
[Timestamp]: [Consolidated Environment Description][Consolidated Summary of Animals and Activities]
"""

PROMPTS["summarize_description_traffic"] = """
You have been provided with a series of chronological descriptions, each detailing a consecutive segment of traffic camera footage from a fixed position. These descriptions were generated from individual video clips, and each follows the format:

[Timestamp]: [Summary of vehicle types, quantities, characteristics, actions, pedestrian activity, and traffic anomalies].

Your task is to act as a **Description Synthesizer and Summarizer** for traffic footage. Your goal is to **merge and consolidate** these multiple descriptions into a single, coherent summary that covers the entire duration of the input segments.

Focus on the following objectives:
1.  **Consolidate Traffic Elements:** Combine all observed traffic elements (vehicle types, quantities, characteristics, actions, pedestrian activity, traffic anomalies) from all input segments into a single, chronologically ordered summary of activity. **Crucially, retain *all* unique information and distinct observations mentioned in *any* of the input descriptions.**
2.  **Eliminate Redundancy:** Remove repetitive phrasing or descriptions of prolonged periods with no change or activity, while ensuring all unique events and states are captured in the consolidated summary.
3.  **Maintain Chronology:** Ensure the consolidated summary of traffic activity flows logically according to the sequence of events across the entire merged timeframe, using timestamps (or relative timing inferred from timestamps) to denote key moments if necessary within the activity summary.

Constraints:
-   Output **a single consolidated description block** following the exact format specified below.
-   Do not introduce information not present in the input descriptions.
-   Do not speculate or infer.
-   Focus the summary on the progression of dynamic events, especially traffic activity and anomalies.
-   Ensure *every* distinct observation from the input is represented in the final consolidated summary.
-   The content within the output fields should be a concise, fact-based summary, with length appropriate to the events observed.

Input: {inputs}

Output Format:
The final output should be a concise, fact-based summary of the traffic activity with the following format:
[Timestamp]: [Consolidated Summary of Traffic Elements (vehicle types, quantities, characteristics, actions, pedestrian activity, traffic anomalies)].
"""

PROMPTS["filter_description"] = """
You are an expert in video scene understanding and grounding natural language to tracked objects.

You are given four inputs:
1) Query: a short phrase describing what the user is asking about (e.g., "the woman", "the person walking", "the baby").
2) Description: a full free-text scene description.
3) Tracks: a JSON list of tracked objects, each with:
   - track_id (int)
   - class (string, e.g., "person", "car", "dog")
   - boxes: a list of detections for that track across time, each item having:
       * frame_number (int), with the corresponding frame number in the given frames sequence.
       * bbox [x_min, y_min, x_max, y_max] in pixel coordinates (top-left, bottom-right), this is normalized to 1000x1000 pixels.
4) Frames: a list of images where all detected or tracked objects are already annotated with bounding boxes labeled as ID: <number>” and their class name (e.g., ID: 1, person”).

### Your task:
- Identify which track_id(s) best match the Query by aligning the Description to the Tracks.
- Only select track IDs that exist in the provided Tracks input.

### Matching guidance (apply pragmatically; do NOT explain these rules in the output):
- **Class compatibility:** Prefer tracks whose `class` matches the Query (e.g., "man/woman/person" → class "person"; "car/vehicle" → class "car"/"truck", etc.). Use reasonable synonyms/singular/plural mapping.
- **Action & motion cues:** If the Description/Query mentions actions (e.g., walking, running, sitting, opening, carrying), infer from temporal bbox patterns (movement vs. static size/position changes) and prefer tracks whose motion plausibly fits.
- **Spatial cues (left/right/center/front/back/near/far):** Approximate from bbox center x,y and area across frames. (Left = smaller x; right = larger x; center = mid-range; near = larger area; far = smaller area.)
- **Temporal cues:** If the Description mentions entering/exiting/approaching/stopping, use the sequence of boxes to favor tracks that appear accordingly (e.g., moving from edge inward).
- **Quantity cues:** If Query implies multiple entities ("two people"), return multiple track_ids that best satisfy count + other cues.
- **Salience:** When ambiguous, favor tracks with longer visibility, clearer motion consistent with the Query, and better class match.
- **No hallucination:** Never invent track IDs; only choose from Tracks. If nothing fits, return an empty list.

### Output format constraints:
- Output **only** valid JSON with this exact structure.
- Do **not** include any explanations, commentary, examples, or Markdown fences.
- Output must begin with {{ and end with }} — nothing else.
- `track_ids` must be a JSON array of integers.
- `final_answer` must be a single concise sentence identifying the best-matching object(s).
- `analysis` must be a brief rationale (1–2 sentences) for why those track IDs match.

### Inputs:
Query: {query}
Description: {description}
Tracks (JSON): {tracks_json}

### Output (strict JSON only):
{{ 
  "track_ids": [matching track ids], 
  "final_answer": "<concise answer describing the relevant object(s)>", 
  "analysis": "<brief reasoning explaining why these track ids match the query>" }}

"""

PROMPTS["visual_filter_description"] = """
You are an expert in visual scene understanding and object grounding.

You are given:
1) Query: a short natural language phrase describing what the user is asking about (e.g., "the man in blue", "the car on the right").
2) Image: a single image where all detected or tracked objects are already annotated with bounding boxes labeled as ID: <number>” and their class name (e.g., ID: 1, person”).
3) Description: a free-text summary describing the visual scene context.

### Your task:
- Identify which **track_id(s)** in the image best match the Query, using both the visual information and the contextual description.
- You can directly “see” the boxes and labels drawn on the image — use them as visual anchors.

### Matching guidance (apply visually; do NOT explain these in output):
- **Class match:** Align nouns in the Query (e.g., person, car, dog) to the labeled class near each ID”.
- **Appearance cues:** Use visible properties such as clothing color, object color, or shape.
- **Spatial cues:** Infer left/right/center/top/bottom/near/far from bounding box placement and size.
- **Interaction cues:** If the Query or Description mentions actions or relations (“the person holding the cup”, “the car next to the bike”), visually match based on proximity or orientation.
- **Multiplicity cues:** If multiple entities are requested (“two people”), return multiple relevant track IDs.
- **Confidence:** When uncertain, favor the most visually salient or contextually consistent candidate.
- **No guessing:** Only output track IDs visible and labeled on the image.

### Output format constraints:
- Output **only** valid JSON in the following exact structure.
- Do NOT include explanations, Markdown, or extra text.
- Output must begin with {{ and end with }}.
- `track_ids` must be an array of integers.
- `final_answer` should be a short, direct identification of the chosen object(s).
- `analysis` should be a brief rationale (1–2 sentences) summarizing the visual reasoning.

### Inputs:
Query: {query}
Description: {description}
Image: (contains bounding boxes labeled like ID: 1, person”, ID: 2, car”, etc.)

### Output (strict JSON only):
{{ 
  "track_ids": [matching track ids], 
  "final_answer": "<concise answer describing the identified object(s)>", 
  "analysis": "<brief reasoning based on visible cues and context>" 
}}
"""

PROMPTS["summary_and_answer_augmented"] = """
You are an expert in video scene understanding and grounding natural language to tracked objects.

You are given four inputs:
1) Query: a short phrase describing what the user is asking about (e.g., "the woman", "the person walking", "the baby").
2) Description: a full free-text scene description.
3) Tracks: a JSON list of tracked objects, each with:
   - track_id (int)
   - class (string, e.g., "person", "car", "dog")
   - boxes: a list of detections for that track across time, each item having:
       * frame_number (int), with the corresponding frame number in the given frames sequence.
       * bbox [x_min, y_min, x_max, y_max] in pixel coordinates (top-left, bottom-right), this is normalized to 1000x1000 pixels.
4) Frames: a list of images where all detected or tracked objects are already annotated with bounding boxes labeled as ID: <number>” and their class name (e.g., ID: 1, person”).

### Your task:
- Identify which track_id(s) best match the Query by aligning the Description to the Tracks.
- Only select track IDs that exist in the provided Tracks input.
- Based on your analysis, choose the best matching track(s) and provide the corresponding multiple-choice answer (A, B, C, or D).

### Matching guidance (apply pragmatically; do NOT explain these rules in the output):
- **Class compatibility:** Prefer tracks whose `class` matches the Query (e.g., "man/woman/person" → class "person"; "car/vehicle" → class "car"/"truck", etc.). Use reasonable synonyms/singular/plural mapping.
- **Action & motion cues:** If the Description/Query mentions actions (e.g., walking, running, sitting, opening, carrying), infer from temporal bbox patterns (movement vs. static size/position changes) and prefer tracks whose motion plausibly fits.
- **Spatial cues (left/right/center/front/back/near/far):** Approximate from bbox center x,y and area across frames. (Left = smaller x; right = larger x; center = mid-range; near = larger area; far = smaller area.)
- **Temporal cues:** If the Description mentions entering/exiting/approaching/stopping, use the sequence of boxes to favor tracks that appear accordingly (e.g., moving from edge inward).
- **Quantity cues:** If Query implies multiple entities ("two people"), return multiple track_ids that best satisfy count + other cues.
- **Salience:** When ambiguous, favor tracks with longer visibility, clearer motion consistent with the Query, and better class match.
- **No hallucination:** Never invent track IDs; only choose from Tracks. If nothing fits, return an empty list.

### Output format:
Return your review, reference, and reasoning process in the `Analysis` field and the answer in the `Answer` field. Choose one of the following options based on your analysis:
- A) [First option]
- B) [Second option]
- C) [Third option]
- D) [Fourth option]

### Inputs:
Query: {query}
Description: {description}
Tracks (JSON): {tracks_json}

### Output (strict JSON only):
{{ 
  "Analysis": "<Brief reasoning about how the tracks match the query based on class, action, motion, etc.>", 
  "Answer": "[A, B, C, or D]" 
}}
"""

PROMPTS["generate_event_to_event_description"] = """
You are an expert in video understanding and temporal event analysis. Your task is to analyze video frames from two connected events and generate a detailed description of how these events are related to each other.

### Context:
You are provided with:
1. **User Query**: The question or task that requires understanding the relationship between events.
2. **Event Pair Information**: Details about two events that are connected in a knowledge graph:
   - Event 1: description, frame range, and metadata
   - Event 2: description, frame range, and metadata
   - Connection score: a numerical value indicating the strength of the connection
3. **Video Frames**: A sequence of frames extracted from both events, showing the visual progression from Event 1 to Event 2.

### Your Task:
Analyze the provided video frames and generate a comprehensive description that explains:
1. **Visual Progression**: How the scene transitions from Event 1 to Event 2, including:
   - Changes in objects, people, or entities
   - Spatial movements or transformations
   - Temporal continuity or gaps
   
2. **Causal or Sequential Relationships**: Identify if Event 1 causes, enables, or leads to Event 2, or if they are part of a larger sequence.

3. **Key Visual Connections**: Describe specific visual elements that link the two events:
   - Recurring objects or people
   - Environmental continuity
   - Action sequences or movements

4. **Relevance to Query**: Explain how this event-to-event connection relates to the user's query.

### Guidelines:
- Focus on **observable visual evidence** from the frames. Do not speculate beyond what is visible.
- Describe the **temporal flow** between events clearly.
- Highlight **distinctive visual markers** that connect the events.
- Keep the description concise but comprehensive (approximately 150-250 words).
- If the frames show a clear cause-effect relationship, make it explicit.
- If the connection is weak or unclear, state this honestly.

### Input:
User Query: {user_query}

Event Pair Information:
{event_pair_info}

### Output Format:
Provide your analysis in JSON format:
{{
  "event_connection_description": "<Detailed description of how Event 1 and Event 2 are visually and temporally connected, including progression, relationships, and relevance to the query>",
  "connection_type": "<Type of connection: causal, sequential, simultaneous, or unclear>",
  "visual_evidence": "<Key visual elements that support the connection>",
  "relevance_to_query": "<How this connection helps answer the user query>"
}}
"""

PROMPTS["generate_final_answer_with_e2e"] = """
You are an advanced AI system designed to answer questions based on video content and knowledge graph information. You will receive:
1. A user query/question
2. Knowledge graph information including events, objects, and their relationships
3. Event-to-event connection descriptions generated from visual analysis of video frames

Your task is to synthesize all this information to provide a comprehensive and accurate answer to the user's query.

### Instructions:
1. **Review the User Query**: Carefully understand what the user is asking about.

2. **Analyze Knowledge Graph Information**: Review the provided events, objects, and their relationships. Pay attention to:
   - Event descriptions and their temporal information
   - Objects associated with each event
   - The overall structure and statistics of the knowledge graph

3. **Integrate Event-to-Event Connections**: The event-to-event descriptions were generated by analyzing video frames. These descriptions provide:
   - Visual evidence of how events are connected
   - Temporal progression between events
   - Causal or sequential relationships
   - Relevance indicators to the query
   
   Use these descriptions to understand the deeper relationships between events that may not be fully captured in the graph structure alone.

4. **Synthesize Information**: Combine information from:
   - The knowledge graph (structured events and objects)
   - The event-to-event visual analysis (temporal and causal connections)
   - Your understanding of the query requirements
   
   To form a coherent answer.

5. **Provide Reasoning**: Explain your reasoning process, including:
   - Which events and objects are most relevant to the query
   - How the event-to-event connections inform your answer
   - Any temporal or causal relationships that are important

6. **Generate Answer**: Provide a clear, direct answer to the user's query based on your analysis.

### Input:
User Query: {user_query}

Knowledge Graph Information:
{video_segments}

Event-to-Event Connection Descriptions:
{event_to_event_descriptions}

### Output Format:
Respond in clean JSON format:
{{
  "Analysis": "<Your detailed reasoning process, explaining how you integrated the knowledge graph information and event-to-event connections to answer the query. Include references to specific events, objects, and connections that are relevant.>",
  "Answer": "A" // Or "B", or "C", or "D". Just the single capital letter.
}}
"""
