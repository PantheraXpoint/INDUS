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

PROMPTS["summarize_descriptions"] = """
You are an expert in summarizing video segment descriptions. Your task is to extract segment information from the sequential video segment descriptions and merge them into a single video event description.

### Event Description Guidelines:
- If the content continues the same scene or content, MERGE then and DON'T duplicate the information.
- The tone of the event description should be as if you are directly describing the video event. Provide a comprehensive narrative of the merged events without separating the content into distinct segments or using line breaks to list different aspects.
- Refrain from using phrases such as "At 2.0 seconds...", "By 10.0 seconds...", "The first/last segment", "The second event begins with...", "The final frames of this segment" or ohter time-related words, which will make the event content fragmented.
- Only provide objective information, avoiding subjective interpretations like mood or atmosphere.

### Output Format:
Please provide the response in one continuous paragraph.

segment description in format of `start_time:end_time:description`:
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

PROMPTS["generate_final_answer_with_reasoning"] = """
You are an advanced AI system designed to answer questions based on video content and knowledge graph information. You will receive:
1. A user query/question
2. Knowledge graph information including events, objects, and their relationships
3. Event-to-event connection descriptions generated from visual analysis of video frames
4. Reasoning results from the reasoning process

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

Reasoning Results of supporting questions:
{reasoning_results}

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

PROMPTS["generate_reasoning_answer"] = """
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
"<Your detailed reasoning process, explaining how you integrated the knowledge graph information and event-to-event connections to answer the query. Include references to specific events, objects, and connections that are relevant.>"
"""

PROMPTS["Reasoning"] = """
You are given ONE QA record from the dataset.

Task:
Answer the main query based on "question type", "question", and "plans".
Main query: Complete the following "plans" to answer the "question".

Plans:
1. vqa("question")
2. vqa(["list of at most three distinct questions that support answering the question; there can be no supporting questions that mention 'question type'"]):response

Guidelines for Question type (qatype):
Use ONLY: what, why, how, where, location, counting

Heuristics:
- counting: query asks "how many …" or requests a count (years, floors, people, vehicles, etc.)
- location: asks where the whole scene/video is (e.g., "where is this video taken", "where is this happening")
- where: asks where a specific thing/object/action target is (e.g., "which direction do they turn", "where is the sign", "where is it projected on")
- why: query contains "why"
- how: asks method/process/reaction/feeling/change (often starts with "how", "how did", "how is … feeling")
- otherwise: what

Supporting question rule:
- If qatype is simple "what", output [].
- Otherwise output 1-3 short, distinct supporting questions that help answer the main question.

DO NOT:
- pick an option letter
- explain which option is correct
- include anything beyond the two vqa calls + justification

========================
FEW-SHOT EXAMPLES
========================

Question type: what
Question: What's the weather like?
Answer:
vqa("What's the weather like?")
vqa([]):question type "what" is given which is sufficiently simple to answer, thus empty supporting question list

Question type: what
Question: What shop did the camera wearer pass before the coffee shop called Espresso?
Answer:
vqa("What shop did the camera wearer pass before the coffee shop called Espresso?")
vqa([]):question type "what" is given which is sufficiently simple to answer, thus empty supporting question list

Question type: counting
Question: The monument commemorates its anniversary of how many years since its establishment?
Answer:
vqa("The monument commemorates its anniversary of how many years since its establishment?")
vqa(["what monument or plaque is visible?", "where is the anniversary number written (sign/plaque)?", "count/read the number of years shown."]):question type "counting" is given which requires to count

Question type: where
Question: After the camera wearer sees the shopping centre, which direction do they turn?
Answer:
vqa("After the camera wearer sees the shopping centre, which direction do they turn?")
vqa(["what surrounds the camera wearer at the turning point (street/intersection)?", "what movement does the camera wearer make next?", "is the turn left, right, or straight?"]):question type "where" is given which requires to ask the context of the scene and the direction of movement

Question type: how
Question: While the camera wearer is observing the wall sculpture inside a church, what change occurs in the lighting?
Answer:
vqa("While the camera wearer is observing the wall sculpture inside a church, what change occurs in the lighting?")
vqa(["what does the interior scene look like before the change?", "what does the lighting look like after the change?", "does it get brighter, dimmer, or change color?"]):question type "how" is given which requires to identify the context of the scene and the change that happens

Question type: why
Question: Why does the camera wearer stop walking near the intersection?
Answer:
vqa("Why does the camera wearer stop walking near the intersection?")
vqa(["what happens right before the camera wearer stops?", "what is the camera wearer doing when they stop?", "what surrounds the camera wearer (traffic, signal, crowd, obstacle)?"]):question type "why" is given which requires to ask context of the scene that the subject is in

Question type: location
Question: Where is this happening?
Answer:
vqa("Where is this happening?")
vqa(["where is this place?", "what can be seen in this scene (buildings/landmarks/street features)?", "identify the main objects/signs that indicate the location."]):question type "location" is given which requires to analyze objects in the scene

========================
NOW DO THIS RECORD
========================
Inputs:
- question and options: {question_and_options}

########################
Output (must match exactly):
Question type: <qatype>
Question: <query>
Answer:
vqa("<query>")
vqa(<supporting_question_list>):<one-sentence justification referencing the question type and why the list is empty or not>
"""

PROMPTS["checkframe_and_answer_with_events"] = """
You are an AI assistant specialized in video understanding. Your task is to analyze video frames along with contextual event descriptions to answer a multiple-choice question.

**Event Context:**
The following events describe what happens in the video segments shown in the frames:

{event_descriptions}

**User Query:** {user_query}

**Your Instructions:**

1.  **Review Event Context:** First, read the event descriptions above. These provide narrative context about what occurs in the video.
2.  **Examine Video Frames:** Next, carefully review the provided video frames. Look for visual details that support or extend the event descriptions.
3.  **Integrate Information:** Combine the event context with visual evidence from the frames to build a comprehensive understanding.
4.  **Reason Step-by-Step:** Think through how both the event descriptions and visual details help answer the User Query. Write down this reasoning process. This will be your "Analysis".
5.  **Choose the Best Answer:** Based on your integrated reasoning, select only ONE letter (A, B, C, or D) that correctly answers the User Query. This will be your "Answer".
6.  **Provide Output in JSON Format ONLY:**
    * Your entire response MUST be a single JSON object.
    * Do NOT write any text or explanation before or after the JSON object.
    * The JSON object must have exactly two keys:
        * `"Analysis"`: Your step-by-step reasoning explaining how you used both event descriptions and video frames to answer the User Query.
        * `"Answer"`: The single letter (A, B, C, or D) you chose as the correct answer.

**REMEMBER: Your response MUST strictly follow this JSON structure:**
```json
{{
  "Analysis": "Your detailed step-by-step reasoning, explaining how you integrated event descriptions and video frames to answer the User Query, goes here.",
  "Answer": "A" // Or "B", or "C", or "D". Just the single capital letter.
}}
"""

PROMPTS["uniform_sampling_answer"] = """
You are an expert AI assistant specialized in comprehensive video understanding and analysis. Your role is to carefully examine video frames and answer questions with high accuracy by leveraging your visual reasoning and temporal understanding capabilities.

**Your Capabilities:**
- Visual perception: Identify objects, people, actions, scenes, and spatial relationships
- Temporal reasoning: Understand sequences, causality, and changes over time
- Contextual analysis: Grasp the overall narrative and situational context
- Detail orientation: Notice fine-grained details that may be crucial for answering questions

**Instructions:**
1. **Observe Carefully:** Examine ALL provided video frames sequentially to build a complete understanding
2. **Identify Key Elements:** Note important objects, actions, events, and their temporal relationships
3. **Reason Logically:** Apply step-by-step reasoning to connect observations with the question
4. **Consider Context:** Use the broader video context to disambiguate and validate your answer
5. **Select Confidently:** Choose the single best answer that most accurately addresses the question

**User Query:**
{user_query}

**Output Requirements:**
- Provide comprehensive step-by-step reasoning in the "Analysis" field explaining your observations and logic
- Select exactly ONE answer letter (A, B, C, or D) in the "Answer" field
- Output ONLY valid JSON format, with no additional text before or after

**Output Format (JSON only):**
```json
{{
  "Analysis": "Detailed step-by-step reasoning that explains what you observe in the frames and how it leads to your answer",
  "Answer": "A" // Or "B", or "C", or "D". Just the single capital letter.
}}
```

Remember: Base your answer EXCLUSIVELY on what you observe in the provided video frames. Do not rely on external knowledge unless explicitly required by the question.
"""


#===================================New Prompts===================================

# =============================================================================
# PROMPT 1: Frames/Images Only (Experiments 1 & 2)
# =============================================================================
PROMPTS["frames_only"] = """
You are an expert AI assistant specialized in video understanding. Your role is to carefully examine video frames and answer questions with high accuracy.

**Video Images:**
The following images are extracted from the video in chronological order:

{frame_timestamps}

**Instructions:**
1. **Examine Sequentially:** Review all images in order, noting their timestamps
2. **Identify Key Details:** Pay attention to objects, actions, people, and changes between images
3. **Reason Temporally:** Use timestamps to understand the sequence of events
4. **Answer the Question:** Select the answer best supported by your observations

**User Query:**
{user_query}

**Output Requirements:**
- Provide step-by-step reasoning in the "Analysis" field, referencing specific images by number and timestamp
- Select exactly ONE answer letter (A, B, C, or D) in the "Answer" field
- Output ONLY valid JSON format

**Output Format (JSON only):**
```json
{{
  "Analysis": "Reasoning referencing specific images (e.g., 'Image 5 at 09:30 shows...')",
  "Answer": "A" // Or "B", or "C", or "D". Just the single capital letter.
}}
```
"""


# =============================================================================
# PROMPT 2: Events Only (Experiments 3 & 4 & 5)
# =============================================================================
PROMPTS["events_only"] = """
You are an expert AI assistant specialized in video understanding. Your role is to analyze event descriptions from a video and answer questions with high accuracy.

**Video Event Descriptions:**
The following numbered events describe what happens in the video, in chronological order:

{event_descriptions}

**Instructions:**
1. **Read All Events:** Review the events in order to understand the video's narrative
2. **Identify Relevant Events:** Determine which events contain information needed to answer the question
3. **Reason Temporally:** Consider the sequence and timing of events
4. **Answer the Question:** Select the answer best supported by the event descriptions

**User Query:**
{user_query}

**Output Requirements:**
- Provide step-by-step reasoning in the "Analysis" field, referencing specific events by number and timestamp
- Select exactly ONE answer letter (A, B, C, or D) in the "Answer" field
- Output ONLY valid JSON format

**Output Format (JSON only):**
```json
{{
  "Analysis": "Reasoning referencing specific events (e.g., 'Event 2 at [07:45 - 10:20] describes...')",
  "Answer": "A" // Or "B", or "C", or "D". Just the single capital letter.
}}
```
"""

# =============================================================================
# PROMPT 3: Frames + Events Aligned (Experiments 6 & 7)
# =============================================================================
PROMPTS["frames_and_events_aligned"] = """
You are an expert AI assistant specialized in video understanding. You will analyze video frames alongside event descriptions to answer questions accurately.

**Event Descriptions with Visual Evidence:**
The following numbered events describe what happens in the video. Each event includes timestamps and references to the specific images (frames) that fall within that time range.

{event_descriptions_with_frames}

**How to Use This Information:**
- Each numbered event describes WHAT happens during a time period
- The "Visual Evidence" line shows which images correspond to that event
- Use event descriptions for NARRATIVE CONTEXT
- Examine the referenced images for VISUAL DETAILS that support or extend the descriptions
- Image numbers (e.g., Image 12) correspond to the frames provided in order

**Instructions:**
1. **Read Each Event:** Understand what happens during each time period
2. **Examine Visual Evidence:** For relevant events, look at the referenced images for specific visual details
3. **Integrate Both:** Combine narrative context with visual observations
4. **Answer the Question:** Select the answer best supported by both sources

**User Query:**
{user_query}

**Output Requirements:**
- In your "Analysis", reference specific events by number AND specific images by number when relevant
- Select exactly ONE answer letter (A, B, C, or D) in the "Answer" field
- Output ONLY valid JSON format

**Output Format (JSON only):**
```json
{{
  "Analysis": "Reasoning referencing events and images (e.g., 'Event 1 describes entering the kitchen, and Image 12 confirms this by showing...')",
  "Answer": "A" // Or "B", or "C", or "D". Just the single capital letter.
}}
```
"""