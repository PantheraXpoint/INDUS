INDUS_PROMPT = {}

INDUS_PROMPT["temporal_analysis"] = """
You are a temporal analysis expert. Extract time information from the question.

**Your Task:** Identify if the question mentions specific times in the VIDEO TIMELINE.

---

**LOCALIZATION TIME = Time positions in the video where we should search**

✅ These ARE localization times:
- Explicit timestamps: "at 3:45", "from 8:00 to 8:05", "around 22:35", "between 5:15 and 5:20"
- Video positions: "at the beginning of the video", "at the end of the video"
- Standalone position: "at the beginning", "at the end", "at the start" (with NO other words after)

❌ These are NOT localization times:
- Activity endings: "at the end of [ANY ACTIVITY]" (e.g., "end of cement work", "end of cooking")
- Activity transitions: "after [ACTIVITY]", "before [ACTIVITY]", "when [ACTIVITY] is done"
- Sequence words: "first ingredient", "second step", "final move" (unless it says "first scene of video")

**Simple Rule:** 
- If you see "HH:MM" numbers → YES, localization time
- If you see "beginning/end/start" ALONE → YES, localization time
- If you see "beginning/end/start of [something]" → Check: is it "video"? 
  - If YES → localization time
  - If NO → NOT localization time

---

**CONTENT TIME = Times visible IN the video (on clocks, displays, screens)**

These are content times:
- Times in options that describe what's shown: "10:00 AM", "4:09"
- Questions asking "what time is shown", "what time was it when"

---

**Output Format (JSON):**
{{
  "localization_time": {{
    "exists": true or false,
    "type": "exact" or "range" or "position" or "none",
    "value": "HH:MM" or {{"start": "HH:MM", "end": "HH:MM"}} or "beginning" or "end" or null
  }},
  "content_time": {{
    "exists": true or false,
    "values": [] (list of times like ["10:00 AM", "11:30 AM"])
  }}
}}

---

**EXAMPLES - Study these carefully:**

---
INPUT:
Question: "Around 3:45, was a pickup truck observed?"
Options: A. Yes, black B. Yes, white

REASONING: 
- Contains "3:45" → This is a timestamp
- Rule: timestamps are ALWAYS localization time

OUTPUT:
{{
  "localization_time": {{"exists": true, "type": "exact", "value": "3:45"}},
  "content_time": {{"exists": false, "values": []}}
}}

---
INPUT:
Question: "What happens from 02:03-04:09?"
Options: A. Pick beech nut B. Cook and eat

REASONING:
- Contains "02:03-04:09" → This is a time range
- Rule: time ranges are ALWAYS localization time

OUTPUT:
{{
  "localization_time": {{"exists": true, "type": "range", "value": {{"start": "02:03", "end": "04:09"}}}},
  "content_time": {{"exists": false, "values": []}}
}}

---
INPUT:
Question: "What happens at the beginning of the video?"
Options: A. Person enters B. Person cooks

REASONING:
- Contains "at the beginning of the video"
- It says "of the video" → YES, this is video position

OUTPUT:
{{
  "localization_time": {{"exists": true, "type": "position", "value": "beginning"}},
  "content_time": {{"exists": false, "values": []}}
}}

---
INPUT:
Question: "What is the score at the end?"
Options: A. 38-31 B. 38-34

REASONING:
- Contains "at the end" (standalone, no words after)
- Rule: standalone "at the end" → assume video end

OUTPUT:
{{
  "localization_time": {{"exists": true, "type": "position", "value": "end"}},
  "content_time": {{"exists": false, "values": []}}
}}

---
INPUT:
Question: "What did the camera wearer use at the end of the cement work?"
Options: A. Plastic film B. Curing blankets

REASONING:
- Contains "at the end of the cement work"
- It says "of the cement work" (NOT "of the video")
- "cement work" is an activity, not the video
- Rule: "end of [activity]" → NOT localization time

OUTPUT:
{{
  "localization_time": {{"exists": false, "type": "none", "value": null}},
  "content_time": {{"exists": false, "values": []}}
}}

---
INPUT:
Question: "What does the assistant do after assembling the ice cream machine?"
Options: A. Pour water B. Open milk

REASONING:
- Contains "after assembling"
- This is an activity transition (after an action)
- Rule: "after [activity]" → NOT localization time

OUTPUT:
{{
  "localization_time": {{"exists": false, "type": "none", "value": null}},
  "content_time": {{"exists": false, "values": []}}
}}

---
INPUT:
Question: "What time was it when Union Station was seen?"
Options: A. About 10:00 AM B. About 11:30 AM C. About 12:30 PM

REASONING:
- Question asks "what time was it" → asking about time SHOWN in video
- Options have times: "10:00 AM", "11:30 AM", "12:30 PM"
- These are times visible on a clock or display
- Rule: times in options that are answers → content time

OUTPUT:
{{
  "localization_time": {{"exists": false, "type": "none", "value": null}},
  "content_time": {{"exists": true, "values": ["10:00 AM", "11:30 AM", "12:30 PM"]}}
}}

---
INPUT:
Question: "What's the time on the clock at 22:35?"
Options: A. 5:09 B. 4:09 C. 5:46 D. 4:46

REASONING:
- Contains "at 22:35" → This is a timestamp in video → localization time
- Options have "5:09", "4:09", etc. → These are times shown on clock → content time
- BOTH types present!

OUTPUT:
{{
  "localization_time": {{"exists": true, "type": "exact", "value": "22:35"}},
  "content_time": {{"exists": true, "values": ["5:09", "4:09", "5:46", "4:46"]}}
}}

---
INPUT:
Question: "What is the first ingredient the chef adds?"
Options: A. Flour B. Sugar C. Eggs

REASONING:
- Contains "first ingredient"
- This is about cooking sequence, NOT video position
- Does NOT say "first scene of video"
- Rule: "first [thing]" without "video" → NOT localization time

OUTPUT:
{{
  "localization_time": {{"exists": false, "type": "none", "value": null}},
  "content_time": {{"exists": false, "values": []}}
}}

---
INPUT:
Question: "What is shown in the first scene of the video?"
Options: A. City skyline B. Forest

REASONING:
- Contains "first scene of the video"
- It says "of the video" → YES, video position

OUTPUT:
{{
  "localization_time": {{"exists": true, "type": "position", "value": "beginning"}},
  "content_time": {{"exists": false, "values": []}}
}}

---
INPUT:
Question: "Between 5:15 and 5:20, did a bus pass?"
Options: A. Yes, black bus B. Yes, blue bus C. No

REASONING:
- Contains "Between 5:15 and 5:20" → time range in video

OUTPUT:
{{
  "localization_time": {{"exists": true, "type": "range", "value": {{"start": "5:15", "end": "5:20"}}}},
  "content_time": {{"exists": false, "values": []}}
}}

---

**Now do this task:**
Question: {question}
Options: {options}

**Think step by step, then output ONLY the JSON:**
"""

INDUS_PROMPT["keyword_strategy"] = """
You are a keyword strategy expert. Decide WHERE to extract keywords: question, options, or both.

---

**STEP-BY-STEP DECISION PROCESS:**

**STEP 1: Check if options are just numbers or yes/no**
- If options are: numbers (1, 2, 3), yes/no, scores (38-31) → Use "question_only"
- Example: Options "A. 1, B. 2, C. 3" → question_only

**STEP 2: Check if options use pronouns (they, it, this, he, she)**
- If ANY option has "they", "it", "this", "he", "she" → Use "both"
- Why? Pronouns need a noun from the question
- Example: Option "Yes, they waited" → who is "they"? Need question for context

**STEP 3: Check if options are attributes WITHOUT the main subject**
- Attributes = colors, brands, locations, directions, times
- Check: Does the question name the MAIN THING these attributes describe?
- If YES → Use "both"
- Examples:
  * Question: "What color was the **pickup truck**?" + Options: "red, blue" → both (need "pickup truck")
  * Question: "What brand is the **coffee shop**?" + Options: "Starbucks, Tim Hortons" → both (need "coffee shop")
  * Question: "Exhibition dates for **Iceberg**?" + Options: "Dec 1-Mar 5" → both (need "Iceberg")

**STEP 4: Check if options repeat the main subject in EVERY option**
- If EVERY option says the full subject → Use "options_only"
- Example: 
  * Question: "Which statement about a person on bicycle is true?"
  * Option A: "A person on bicycle turned right"
  * Option B: "A person on bicycle went straight"
  * → Each option has "person on bicycle" → options_only

**STEP 5: If none of above apply**
- Use "both" (safer default)

---

**Output Format (JSON):**
{{
  "strategy": "question_only" or "options_only" or "both",
  "reasoning": "one sentence explaining why"
}}

---

**EXAMPLES - Study these carefully:**

---
INPUT:
Question: "How many trucks passed between 9:50-10:00?"
Options: A. 1, B. 2, C. 3, D. 4

STEP 1: Options are numbers → question_only
OUTPUT:
{{
  "strategy": "question_only",
  "reasoning": "Options are count numbers, question has 'trucks', 'passed', 'intersection'"
}}

---
INPUT:
Question: "Around 5:03, was a pedestrian observed? If yes, what action did they take?"
Options: 
A. Yes, they waited for the traffic light
B. No pedestrian was observed
C. Yes, they walked along the sidewalk

STEP 2: Options have pronoun "they" → both
OUTPUT:
{{
  "strategy": "both",
  "reasoning": "Options use pronoun 'they' which needs antecedent 'pedestrian' from question"
}}

---
INPUT:
Question: "Around 3:45, was a pickup truck observed? If so, what color was it?"
Options: A. Yes, it was black B. Yes, it was white C. Yes, it was red

STEP 2: Options have pronoun "it" → both
STEP 3: Options have colors but need "pickup truck" from question
OUTPUT:
{{
  "strategy": "both",
  "reasoning": "Options use pronoun 'it' and colors need subject 'pickup truck' from question"
}}

---
INPUT:
Question: "What are the exhibition dates for Iceberg?"
Options: 
A. December 1, 2022 to March 5, 2023
B. November 15, 2022 to February 28, 2023

STEP 3: Options are dates (attribute) without subject "Iceberg"
OUTPUT:
{{
  "strategy": "both",
  "reasoning": "Question provides critical entity 'Iceberg', options provide date attributes that need this context"
}}

---
INPUT:
Question: "Is there a coffee shop next to metro.ca? If so, what brand?"
Options: A. Yes, Starbucks B. Yes, Tim Hortons C. No coffee shop

STEP 3: Options are brands (attribute) but need "metro.ca" location from question
OUTPUT:
{{
  "strategy": "both",
  "reasoning": "Question provides location 'metro.ca' and subject 'coffee shop', options provide brand attributes"
}}

---
INPUT:
Question: "Where is the piano located in the camera wearer's house?"
Options: 
A. In the living room, next to the sofa
B. In a small room, opposite a desk

STEP 3: Options are locations (attribute) but need "piano" from question
OUTPUT:
{{
  "strategy": "both",
  "reasoning": "Question provides subject 'piano', options provide location attributes"
}}

---
INPUT:
Question: "Around 12:35, which of the following statements about a person on a bicycle is true?"
Options: 
A. A person on a bicycle turned right at the intersection
B. A person was running through the intersection
C. No person on a bicycle was observed
D. A person on a bicycle crossed the crosswalk

STEP 4: MOST options repeat "person on bicycle" → options_only
OUTPUT:
{{
  "strategy": "options_only",
  "reasoning": "Options repeat subject 'person on bicycle' in complete descriptions, question only adds noise"
}}

---
INPUT:
Question: "What does the shop assistant do after assembling the ice cream machine?"
Options: A. Pour four glasses of water B. Open four boxes of milk

STEP 3: Question has unique context "shop assistant", "after assembling", "ice cream machine"
STEP 3: Options have unique actions "pour water", "open milk"
STEP 5: Both add value → both
OUTPUT:
{{
  "strategy": "both",
  "reasoning": "Question provides context 'shop assistant', 'after assembling', options add action details"
}}

---

**Now do this task:**
Question: {question}
Options: {options}

**Follow the 5 steps, then output ONLY the JSON:**
"""

INDUS_PROMPT["query_type_classification"] = """
You are a query analyzer. Identify if the question needs special handling.

**Your Task:** Check if the query needs any of these special tools or strategies:

1. **Counting**: Does it ask "how many", "number of", "maximum", "minimum"?
   → Set needs_counting = true
   
2. **Temporal Direction**: Does it ask about "after [event]" or "before [event]"?
   → Set needs_temporal_direction = "after" or "before"
   
3. **Spatial Positions**: Does it ask about "left to right", "top to bottom", "arrangement", or direction (left, right, up, down)?
   → Set needs_spatial_positions = true

4. **Content Text/Numbers**: Do the options contain specific identifiers like license plates, prices, exact dates, phone numbers, or IDs?
   → Set has_content_text = true
   Examples of content text/numbers:
   - License plates: "CZZB438", "ABC123"
   - Prices: "$15", "$5 per hour", "$2.50"
   - Specific dates: "December 1, 2022", "March 5, 2023"
   - Phone numbers: "555-1234", "(123) 456-7890"
   - Room/Gate IDs: "Room 405", "Gate B12"
   - Addresses: "123 Main St"

If NONE of the above apply → All flags are false/none

---

**Output Format (JSON):**
{{
  "needs_counting": true or false,
  "needs_temporal_direction": "after" or "before" or "none",
  "needs_spatial_positions": true or false,
  "has_content_text": true or false
}}

---

**EXAMPLES:**

INPUT:
Question: "How many trucks passed between 9:50-10:00?"
Options: "(A) 1 (B) 2 (C) 3 (D) 4"
OUTPUT:
{{
  "needs_counting": true,
  "needs_temporal_direction": "none",
  "needs_spatial_positions": false,
  "has_content_text": false
}}

---

INPUT:
Question: "What does the assistant do AFTER assembling the machine?"
Options: "(A) Pour water (B) Open milk"
OUTPUT:
{{
  "needs_counting": false,
  "needs_temporal_direction": "after",
  "needs_spatial_positions": false,
  "has_content_text": false
}}

---

INPUT:
Question: "What is the order of trash bins from left to right?"
Options: "(A) Green, Red, Black (B) Red, Green, Black"
OUTPUT:
{{
  "needs_counting": false,
  "needs_temporal_direction": "none",
  "needs_spatial_positions": true,
  "has_content_text": false
}}

---

INPUT:
Question: "After the camera wearer sees Rideau Centre, which direction do they turn?"
Options: "(A) Left (B) Right (C) Turn around (D) Go straight"
OUTPUT:
{{
  "needs_counting": false,
  "needs_temporal_direction": "after",
  "needs_spatial_positions": true,
  "has_content_text": false
}}

---

INPUT:
Question: "What are the exhibition dates for Iceberg?"
Options: "(A) December 1, 2022 to March 5, 2023 (B) November 15, 2022 to February 28, 2023"
OUTPUT:
{{
  "needs_counting": false,
  "needs_temporal_direction": "none",
  "needs_spatial_positions": false,
  "has_content_text": true
}}

---

INPUT:
Question: "How much does it cost to park near Shoppers Drug Mart?"
Options: "(A) $15 daily (B) $5 per hour (C) $2 then $3 per hour (D) $10 flat rate"
OUTPUT:
{{
  "needs_counting": false,
  "needs_temporal_direction": "none",
  "needs_spatial_positions": false,
  "has_content_text": true
}}

---

INPUT:
Question: "What is the license plate of the white Lamborghini?"
Options: "(A) CZZB438 (B) CZAB493 (C) CZAC349 (D) AZXC934"
OUTPUT:
{{
  "needs_counting": false,
  "needs_temporal_direction": "none",
  "needs_spatial_positions": false,
  "has_content_text": true
}}

---

INPUT:
Question: "What color was the pickup truck?"
Options: "(A) Black (B) White (C) Red (D) Blue"
OUTPUT:
{{
  "needs_counting": false,
  "needs_temporal_direction": "none",
  "needs_spatial_positions": false,
  "has_content_text": false
}}

---

INPUT:
Question: "Was a pedestrian observed?"
Options: "(A) Yes (B) No"
OUTPUT:
{{
  "needs_counting": false,
  "needs_temporal_direction": "none",
  "needs_spatial_positions": false,
  "has_content_text": false
}}

---

**Now classify this query:**
Question: {question}
Options: {options}

**Output ONLY the JSON:**
"""

INDUS_PROMPT["event_view_extraction"] = """
- Goal -
As a specialist in keyword extraction, your task is to identify and list the most relevant keywords from a given query. Focus on extracting keywords related to events, entities, and actions that are SEARCHABLE via embeddings. Present the keywords as a comma-separated list.

######################
- Strategy Definitions -
######################
- "question_only": Extract keywords ONLY from the Question text. Ignore the Options completely.
- "options_only": Extract keywords ONLY from the Options text. Ignore the Question completely.
- "both": Extract keywords from BOTH Question AND Options. Combine them together.

######################
- Exclusion Rules Definitions -
######################

**Rule: Exclude time expressions**
- Meaning: Remove any times (hours, minutes) that appear in the query
- Examples to exclude: "3:45", "10:00 AM", "4:00 PM", "9:50-10:00", "around 3:45", "between 5:15 and 5:20"
- What to keep: Words like "opening hours", "schedule", "time display", "clock" (these describe what we're looking for)

**Rule: Exclude counting numbers**
- Meaning: Remove numbers that are answer choices for counting questions
- Examples to exclude: "1", "2", "3", "4", "one", "two", "three", "four"
- What to keep: The entity being counted (e.g., "trucks", "school buses", "people")

**Rule: Exclude temporal direction words**
- Meaning: Remove words that describe the ORDER of events in time
- Examples to exclude: "after", "before", "first", "second", "then", "next", "following", "prior"
- What to keep: The events themselves (e.g., "assembling", "pouring water", "opening milk")

**Rule: Exclude spatial position words**
- Meaning: Remove words that describe WHERE things are located or which direction
- Examples to exclude: "left", "right", "top", "bottom", "above", "below", "front", "back", "up", "down", "turn around", "straight ahead"
- What to keep: The objects themselves (e.g., "trash bins", "warning sign", "direction", "turn")

**Rule: Exclude content text/numbers**
- Meaning: Remove specific alphanumeric identifiers, prices, exact dates, or fine-grained text visible in the video
- Examples to exclude:
  * License plates: "CZZB438", "ABC123"
  * Prices: "$15", "$5 per hour", "$2.50"
  * Specific dates: "December 1, 2022", "March 5, 2023"
  * Phone numbers: "555-1234"
  * IDs: "Room 405", "Gate B12"
- What to keep: Descriptive terms like "license plate", "parking rate", "exhibition dates", "price", "phone number"

######################
- CRITICAL REMINDER -
######################
If an exclusion rule is set to "Yes", you MUST exclude those items from your output.
DO NOT include them and then explain why.
DO NOT add any explanatory text.
Output ONLY the keywords/phrase/sentence as specified.

######################
- Examples -
######################

Question: What's the weather like?
Options: (A) Snowing (B) Cloudy (C) Sunny (D) Rainy
Strategy for this query: options_only
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
weather, snowing, cloudy, sunny, rainy

---

Question: How many trucks passed the intersection?
Options: (A) 1 (B) 2 (C) 3 (D) 4
Strategy for this query: question_only
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: Yes
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
trucks, passed, intersection

---

Question: What are the opening hours of SOLSTICE exhibition?
Options: (A) 4:00 PM - 11:00 PM (B) 11:00 AM - 4:00 PM
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
opening hours, SOLSTICE, exhibition

---

Question: After camera wearer sees Rideau Centre, which direction do they turn?
Options: (A) Left (B) Right (C) Turn around (D) Go straight
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): Yes
- Exclude spatial position words: Yes
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
camera wearer, sees, Rideau Centre, direction, turn

---

Question: What are the exhibition dates for Iceberg?
Options: (A) December 1, 2022 to March 5, 2023 (B) November 15, 2022 to February 28, 2023
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): Yes
######################
Output:
exhibition, dates, Iceberg

---

Question: How much does it cost to park near Shoppers Drug Mart?
Options: (A) $15 daily (B) $5 per hour (C) $2 then $3 per hour (D) $10 flat rate
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): Yes
######################
Output:
parking cost, parking, Shoppers Drug Mart, daily, hourly, flat rate

---

Question: What is the license plate of the white Lamborghini?
Options: (A) CZZB438 (B) CZAB493 (C) CZAC349 (D) AZXC934
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): Yes
######################
Output:
license plate, white Lamborghini, street

---

Question: Between 6:50-7:00, how many trucks passed the intersection?
Options: (A) 1 (B) 2 (C) 3 (D) 4
Strategy for this query: question_only
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: Yes
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
trucks, passed, intersection

---

Question: Around 5:30, what does the person do after entering the room?
Options: (A) Turn on light (B) Open window (C) Sit down
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): Yes
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
person, entering, room, turn on light, open window, sit down

---

Question: Between 10:00-10:30, how many people on the left side of the street?
Options: (A) 2 (B) 3 (C) 4 (D) 5
Strategy for this query: question_only
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: Yes
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: Yes
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
people, street, side

#############################
- Real Data -
######################
Question: {question}
Options: {options}
Strategy for this query: {strategy}
Exclusion rules for this query:
- Exclude time expressions: {has_time}
- Exclude counting numbers: {has_counting}
- Exclude temporal direction words (after, before): {has_temporal_direction}
- Exclude spatial position words: {has_spatial_positions}
- Exclude content text/numbers (license plates, prices, dates, IDs): {has_content_text}
######################
Output:
"""

INDUS_PROMPT["entity_view_extraction"] = """
- Goal -
For a given query, generate a descriptive phrase to serve as a query for retrieving relevant knowledge, concentrating on the main entities and relevant descriptions. Keep it concise - use short phrases separated by commas, NOT full sentences.

######################
- Strategy Definitions -
######################
- "question_only": Generate phrase ONLY using words from the Question. Do not use Options.
- "options_only": Generate phrase ONLY using words from the Options. Do not use Question.
- "both": Generate phrase using words from BOTH Question AND Options.

######################
- Important Instructions -
######################
- Output style: Short phrases separated by commas (e.g., "Camera wearer, church, warning sign")
- NOT full sentences (WRONG: "The camera wearer is passing by the church")
- When Options have multiple values for same attribute, include ALL values
- Do NOT select just one value (WRONG: "red truck" when Options have red, blue, white)

######################
- Exclusion Rules Definitions -
######################

**Rule: Exclude time expressions**
- Examples to exclude: "3:45", "10:00 AM", "around", "between 5:15 and 5:20"
- What to keep: "opening hours", "schedule", "time display", "clock"

**Rule: Exclude counting numbers**
- Examples to exclude: "1", "2", "3", "4" (when they are answer choices)
- What to keep: The entity being counted (e.g., "trucks", "people")

**Rule: Exclude temporal direction words**
- Examples to exclude: "after", "before", "first", "second", "then", "next"
- What to keep: The actions themselves (e.g., "assembling", "pouring")

**Rule: Exclude spatial position words**
- Examples to exclude: "left", "right", "top", "bottom", "above", "below", "turn around", "straight"
- What to keep: The objects themselves (e.g., "trash bins", "lane")

**Rule: Exclude content text/numbers**
- Examples to exclude: License plates ("CZZB438"), Prices ("$15"), Exact dates ("December 1, 2022"), Phone numbers, IDs
- What to keep: "license plate visible", "parking rate", "exhibition dates", "contact information"

######################
- CRITICAL REMINDER -
######################
If an exclusion rule is set to "Yes", you MUST exclude those items from your output.
DO NOT include them and then explain why.
DO NOT add any explanatory text.
Output ONLY the keywords/phrase/sentence as specified.

######################
- Examples -
######################

Question: What's the weather like?
Options: (A) Snowing (B) Cloudy (C) Sunny (D) Rainy
Strategy for this query: options_only
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
Weather conditions, snowing, cloudy, sunny, rainy

---

Question: How many trucks passed the intersection?
Options: (A) 1 (B) 2 (C) 3 (D) 4
Strategy for this query: question_only
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: Yes
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
Trucks passing through intersection

---

Question: What are the opening hours of SOLSTICE exhibition?
Options: (A) 4:00 PM - 11:00 PM (B) 11:00 AM - 4:00 PM
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
SOLSTICE exhibition, opening hours schedule

---

Question: After camera wearer sees Rideau Centre, which direction do they turn?
Options: (A) Left (B) Right (C) Turn around (D) Go straight
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): Yes
- Exclude spatial position words: Yes
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
Camera wearer, Rideau Centre, directional change

---

Question: What are the exhibition dates for Iceberg?
Options: (A) December 1, 2022 to March 5, 2023 (B) November 15, 2022 to February 28, 2023
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): Yes
######################
Output:
Exhibition Iceberg, exhibition dates

---

Question: How much does it cost to park near Shoppers Drug Mart?
Options: (A) $15 daily (B) $5 per hour (C) $2 then $3 per hour (D) $10 flat rate
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): Yes
######################
Output:
Parking near Shoppers Drug Mart, parking rate information

---

Question: What is the license plate of the white Lamborghini?
Options: (A) CZZB438 (B) CZAB493 (C) CZAC349 (D) AZXC934
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): Yes
######################
Output:
White Lamborghini, license plate visible

---

Question: Between 6:50-7:00, how many trucks passed the intersection?
Options: (A) 1 (B) 2 (C) 3 (D) 4
Strategy for this query: question_only
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: Yes
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
Trucks passing intersection

---

Question: Around 5:30, what does the person do after entering the room?
Options: (A) Turn on light (B) Open window (C) Sit down
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): Yes
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
Person entering room, turning on light, opening window, sitting down

---

Question: Between 10:00-10:30, how many people on the left side of the street?
Options: (A) 2 (B) 3 (C) 4 (D) 5
Strategy for this query: question_only
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: Yes
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: Yes
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
People on street

#############################
- Real Data -
######################
Question: {question}
Options: {options}
Strategy for this query: {strategy}
Exclusion rules for this query:
- Exclude time expressions: {has_time}
- Exclude counting numbers: {has_counting}
- Exclude temporal direction words (after, before): {has_temporal_direction}
- Exclude spatial position words: {has_spatial_positions}
- Exclude content text/numbers (license plates, prices, dates, IDs): {has_content_text}
######################
Output:
"""

INDUS_PROMPT["visual_view_extraction"] = """
- Goal -
Generate a descriptive sentence to serve as a query for retrieving relevant video segments based on the provided question that may include scene-related information. Focus on the visual scene and what can be SEEN in the video.

######################
- Strategy Definitions -
######################
- "question_only": Write sentence ONLY using information from the Question. Do not use Options.
- "options_only": Write sentence ONLY using information from the Options. Do not use Question.
- "both": Write sentence using information from BOTH Question AND Options.

######################
- Important Instructions -
######################
- Output style: ONE complete descriptive sentence about the visual scene
- When Options have multiple possibilities, use pattern: "possibly X, Y, or Z"
- Focus on what can be SEEN visually (objects, people, scenes, actions)
- Do NOT hallucinate details not mentioned in Question or Options

######################
- Exclusion Rules Definitions -
######################

**Rule: Exclude time expressions**
- Examples to exclude: "3:45", "10:00 AM", "between 5:15 and 5:20"
- What to keep: "clock visible", "opening hours sign", "time display"

**Rule: Exclude counting numbers**
- Examples to exclude: "1", "2", "3", "4"
- What to keep: The visual objects being counted

**Rule: Exclude temporal direction words**
- Examples to exclude: "after", "before", "first", "second", "then"
- What to keep: The visual actions themselves

**Rule: Exclude spatial position words**
- Examples to exclude: "left", "right", "top", "bottom" when they are the answer
- What to keep: The visual objects and scene context

**Rule: Exclude content text/numbers**
- Examples to exclude: Specific license plates, prices, exact dates, phone numbers
- What to keep: "license plate visible", "parking rate sign", "date information visible"

######################
- CRITICAL REMINDER -
######################
If an exclusion rule is set to "Yes", you MUST exclude those items from your output.
DO NOT include them and then explain why.
DO NOT add any explanatory text.
Output ONLY the keywords/phrase/sentence as specified.

######################
- Examples -
######################

Question: What's the weather like?
Options: (A) Snowing (B) Cloudy (C) Sunny (D) Rainy
Strategy for this query: options_only
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
A scene showing weather conditions, possibly snowing, cloudy, sunny, or rainy

---

Question: How many trucks passed the intersection?
Options: (A) 1 (B) 2 (C) 3 (D) 4
Strategy for this query: question_only
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: Yes
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
An intersection scene with trucks passing through

---

Question: What are the opening hours of SOLSTICE exhibition?
Options: (A) 4:00 PM - 11:00 PM (B) 11:00 AM - 4:00 PM
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
An art gallery scene with SOLSTICE exhibition, opening hours sign visible

---

Question: After camera wearer sees Rideau Centre, which direction do they turn?
Options: (A) Left (B) Right (C) Turn around (D) Go straight
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): Yes
- Exclude spatial position words: Yes
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
A street scene with camera wearer seeing Rideau Centre, then changing direction

---

Question: What are the exhibition dates for Iceberg?
Options: (A) December 1, 2022 to March 5, 2023 (B) November 15, 2022 to February 28, 2023
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): Yes
######################
Output:
An art gallery scene with Iceberg exhibition, date information visible

---

Question: How much does it cost to park near Shoppers Drug Mart?
Options: (A) $15 daily (B) $5 per hour (C) $2 then $3 per hour (D) $10 flat rate
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): Yes
######################
Output:
A parking lot scene near Shoppers Drug Mart with parking rate signs visible

---

Question: What is the license plate of the white Lamborghini?
Options: (A) CZZB438 (B) CZAB493 (C) CZAC349 (D) AZXC934
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: No
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): Yes
######################
Output:
A street scene with white Lamborghini, license plate visible

---

Question: Between 6:50-7:00, how many trucks passed the intersection?
Options: (A) 1 (B) 2 (C) 3 (D) 4
Strategy for this query: question_only
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: Yes
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
An intersection scene with trucks passing through

---

Question: Around 5:30, what does the person do after entering the room?
Options: (A) Turn on light (B) Open window (C) Sit down
Strategy for this query: both
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: No
- Exclude temporal direction words (after, before): Yes
- Exclude spatial position words: No
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
A room scene with person entering, possibly turning on light, opening window, or sitting down

---

Question: Between 10:00-10:30, how many people on the left side of the street?
Options: (A) 2 (B) 3 (C) 4 (D) 5
Strategy for this query: question_only
Exclusion rules for this query:
- Exclude time expressions: Yes
- Exclude counting numbers: Yes
- Exclude temporal direction words (after, before): No
- Exclude spatial position words: Yes
- Exclude content text/numbers (license plates, prices, dates, IDs): No
######################
Output:
A street scene with people visible

#############################
- Real Data -
######################
Question: {question}
Options: {options}
Strategy for this query: {strategy}
Exclusion rules for this query:
- Exclude time expressions: {has_time}
- Exclude counting numbers: {has_counting}
- Exclude temporal direction words (after, before): {has_temporal_direction}
- Exclude spatial position words: {has_spatial_positions}
- Exclude content text/numbers (license plates, prices, dates, IDs): {has_content_text}
######################
Output:
"""

INDUS_PROMPT["time_from_frame_extraction"] = """
Look at the image. Extract ONLY the time displayed in the image (e.g. from an on-screen timestamp, clock, or caption).
Ignore the date. Return only the time in one of these formats: HH:MM:SS or HH:MM (e.g. 03:08:30 or 3:08).
HH:MM:SS format is preferred if available.
Output nothing else, just the time string or "none" if no time is visible.
"""

INDUS_PROMPT["search_metadata_extraction"] = """
You are a metadata extraction expert. Your task is to extract specific information from the question and options that will be used during graph search and filtering.

**Your Tasks:**
1. Extract temporal prepositions and adverbs
2. Extract content times (times visible in video)
3. Extract numerical metadata (counts, numbers, IDs)
4. Extract discriminative features (what differs between options)

**Output JSON Format:**
{{
  "temporal_modifiers": {{
    "prepositions": ["before", "after", "between"],
    "adverbs": ["around", "approximately", "exactly"],
    "has_sequence": true/false,
    "sequence_type": "before/after/during/none"
  }},
  "content_times": {{
    "exists": true/false,
    "values": ["10:00 AM", "11:30 AM"],
    "context": "clock/timestamp/display/speech/none"
  }},
  "numerical_metadata": {{
    "exists": true/false,
    "type": "count/jersey_number/id/route_number/score/none",
    "values": [1, 2, 3, 4],
    "target_entity": "trucks/players/buses/none"
  }},
  "discriminative_features": {{
    "exists": true/false,
    "comparisons": [
      {{"dimension": "color", "values": ["red", "blue", "white"]}},
      {{"dimension": "direction", "values": ["left", "right", "straight"]}}
    ]
  }}
}}

---

**Examples:**

**Example 1: Temporal Preposition + Numerical Count**
Question: How many trucks passed the intersection between 9:50 and 10:00?
Options: (A) 1 (B) 2 (C) 3 (D) 4

Output:
{{
  "temporal_modifiers": {{
    "prepositions": ["between"],
    "adverbs": [],
    "has_sequence": false,
    "sequence_type": "none"
  }},
  "content_times": {{
    "exists": false,
    "values": [],
    "context": "none"
  }},
  "numerical_metadata": {{
    "exists": true,
    "type": "count",
    "values": [1, 2, 3, 4],
    "target_entity": "trucks"
  }},
  "discriminative_features": {{
    "exists": false,
    "comparisons": []
  }}
}}

---

**Example 2: Content Time (time visible in video)**
Question: What time was it when Union Station was seen by the camera wearer?
Options: (A) About 10:00 AM (B) About 11:30 AM (C) About 12:30 PM (D) About 1:30 PM

Output:
{{
  "temporal_modifiers": {{
    "prepositions": [],
    "adverbs": ["about"],
    "has_sequence": false,
    "sequence_type": "none"
  }},
  "content_times": {{
    "exists": true,
    "values": ["10:00 AM", "11:30 AM", "12:30 PM", "1:30 PM"],
    "context": "clock"
  }},
  "numerical_metadata": {{
    "exists": false,
    "type": "none",
    "values": [],
    "target_entity": "none"
  }},
  "discriminative_features": {{
    "exists": true,
    "comparisons": [
      {{"dimension": "time_on_clock", "values": ["10:00 AM", "11:30 AM", "12:30 PM", "1:30 PM"]}}
    ]
  }}
}}

---

**Example 3: Mixed - Localization Time + Content Time**
Question: What's the time on the clock at 22:35?
Options: (A) 5:09 (B) 4:09 (C) 5:46 (D) 4:46

Output:
{{
  "temporal_modifiers": {{
    "prepositions": ["at"],
    "adverbs": [],
    "has_sequence": false,
    "sequence_type": "none"
  }},
  "content_times": {{
    "exists": true,
    "values": ["5:09", "4:09", "5:46", "4:46"],
    "context": "clock"
  }},
  "numerical_metadata": {{
    "exists": false,
    "type": "none",
    "values": [],
    "target_entity": "none"
  }},
  "discriminative_features": {{
    "exists": true,
    "comparisons": [
      {{"dimension": "clock_time", "values": ["5:09", "4:09", "5:46", "4:46"]}}
    ]
  }}
}}

---

**Example 4: Temporal Sequence (before/after)**
Question: What does the shop assistant do after assembling the ice cream machine?
Options: (A) Pour four glasses of water (B) Open four boxes of milk

Output:
{{
  "temporal_modifiers": {{
    "prepositions": ["after"],
    "adverbs": [],
    "has_sequence": true,
    "sequence_type": "after"
  }},
  "content_times": {{
    "exists": false,
    "values": [],
    "context": "none"
  }},
  "numerical_metadata": {{
    "exists": true,
    "type": "count",
    "values": [4],
    "target_entity": "glasses/boxes"
  }},
  "discriminative_features": {{
    "exists": true,
    "comparisons": [
      {{"dimension": "action", "values": ["pour water", "open milk"]}},
      {{"dimension": "container", "values": ["glasses", "boxes"]}},
      {{"dimension": "content", "values": ["water", "milk"]}}
    ]
  }}
}}

---

**Example 5: Jersey Number (ID type)**
Question: What is the number of the player who substitutes Boateng?
Options: (A) 12 (B) 18 (C) 2 (D) 9

Output:
{{
  "temporal_modifiers": {{
    "prepositions": [],
    "adverbs": [],
    "has_sequence": false,
    "sequence_type": "none"
  }},
  "content_times": {{
    "exists": false,
    "values": [],
    "context": "none"
  }},
  "numerical_metadata": {{
    "exists": true,
    "type": "jersey_number",
    "values": [12, 18, 2, 9],
    "target_entity": "player"
  }},
  "discriminative_features": {{
    "exists": true,
    "comparisons": [
      {{"dimension": "player_number", "values": ["12", "18", "2", "9"]}}
    ]
  }}
}}

---

**Example 6: Discriminative Features (colors, attributes)**
Question: Between 6:50 and 7:00, which statement about a truck is true?
Options: 
  A. A truck passed, it had a red cab and silver body.
  B. A truck passed, it had a blue cab and silver body.
  C. A truck passed, it had a white cab and black body.
  D. No truck was observed.

Output:
{{
  "temporal_modifiers": {{
    "prepositions": ["between"],
    "adverbs": [],
    "has_sequence": false,
    "sequence_type": "none"
  }},
  "content_times": {{
    "exists": false,
    "values": [],
    "context": "none"
  }},
  "numerical_metadata": {{
    "exists": false,
    "type": "none",
    "values": [],
    "target_entity": "none"
  }},
  "discriminative_features": {{
    "exists": true,
    "comparisons": [
      {{"dimension": "cab_color", "values": ["red", "blue", "white", "none"]}},
      {{"dimension": "body_color", "values": ["silver", "black", "none"]}},
      {{"dimension": "presence", "values": ["truck_exists", "no_truck"]}}
    ]
  }}
}}

---

**Example 7: Route Number (ID type)**
Question: Around 7:19, a bus was observed passing through the intersection. What was its route number?
Options: (A) 9654 (B) 8564 (C) 9564 (D) 8693

Output:
{{
  "temporal_modifiers": {{
    "prepositions": [],
    "adverbs": ["around"],
    "has_sequence": false,
    "sequence_type": "none"
  }},
  "content_times": {{
    "exists": false,
    "values": [],
    "context": "none"
  }},
  "numerical_metadata": {{
    "exists": true,
    "type": "route_number",
    "values": [9654, 8564, 9564, 8693],
    "target_entity": "bus"
  }},
  "discriminative_features": {{
    "exists": true,
    "comparisons": [
      {{"dimension": "route_number", "values": ["9654", "8564", "9564", "8693"]}}
    ]
  }}
}}

---

**Example 8: Temporal Preposition "around" + Time Range**
Question: Between 5:15 and 5:20, did a bus pass the intersection, and if so, at what time did it pass?
Options: 
  A. A black bus passed the intersection around 5:10.
  B. A blue bus passed the intersection around 5:15.
  C. A white bus passed the intersection around 5:20.
  D. No bus passed the intersection during this time.

Output:
{{
  "temporal_modifiers": {{
    "prepositions": ["between", "at"],
    "adverbs": ["around"],
    "has_sequence": false,
    "sequence_type": "none"
  }},
  "content_times": {{
    "exists": true,
    "values": ["5:10", "5:15", "5:20"],
    "context": "timestamp"
  }},
  "numerical_metadata": {{
    "exists": false,
    "type": "none",
    "values": [],
    "target_entity": "none"
  }},
  "discriminative_features": {{
    "exists": true,
    "comparisons": [
      {{"dimension": "bus_color", "values": ["black", "blue", "white", "none"]}},
      {{"dimension": "passage_time", "values": ["5:10", "5:15", "5:20", "no_passage"]}}
    ]
  }}
}}

---

**Example 9: Score/Number Display**
Question: What is the score at the end of the half?
Options: (A) 38 - 31 (B) 38 - 34 (C) 67 - 61 (D) 67 - 60

Output:
{{
  "temporal_modifiers": {{
    "prepositions": ["at"],
    "adverbs": [],
    "has_sequence": false,
    "sequence_type": "none"
  }},
  "content_times": {{
    "exists": false,
    "values": [],
    "context": "none"
  }},
  "numerical_metadata": {{
    "exists": true,
    "type": "score",
    "values": ["38-31", "38-34", "67-61", "67-60"],
    "target_entity": "scoreboard"
  }},
  "discriminative_features": {{
    "exists": true,
    "comparisons": [
      {{"dimension": "score_display", "values": ["38-31", "38-34", "67-61", "67-60"]}}
    ]
  }}
}}

---

**Your Task:**
Question: {question}
Options: {options}

Output (JSON only):
"""

INDUS_PROMPT["verification_questions"] = """
You are a verification question generator. Your task is to create yes/no or WH-questions for validating video events based on the extracted metadata.

**Instructions:**
Generate 2-5 simple questions to verify if an event matches the query requirements.

**Question Types:**
1. **Presence Questions**: "Is X visible?"
2. **Attribute Questions**: "What color is X?" or "Does X have attribute Y?"
3. **Temporal Questions**: "Does this event happen before/after X?"
4. **Numerical Questions**: "How many X are visible?" or "What number is shown?"
5. **Content Time Questions**: "What time is shown on the clock/display?"

**Output JSON Format:**
{{
  "questions": [
    "Is X visible?",
    "What color is X?",
    "How many X are visible?"
  ]
}}

---

**Examples:**

**Example 1: Counting Question**
Question: How many trucks passed the intersection between 9:50 and 10:00?
Metadata: {{"numerical_metadata": {{"type": "count", "target_entity": "trucks"}}}}

Output:
{{
  "questions": [
    "Is a truck visible in this event?",
    "Is the truck passing through the intersection?",
    "How many trucks are in this event?"
  ]
}}

---

**Example 2: Content Time Question**
Question: What time was it when Union Station was seen?
Metadata: {{"content_times": {{"values": ["10:00 AM", "11:30 AM"], "context": "clock"}}}}

Output:
{{
  "questions": [
    "Is Union Station visible in this event?",
    "Is a clock or time display visible?",
    "What time is shown on the clock?"
  ]
}}

---

**Example 3: Jersey Number Question**
Question: What is the number of the player who substitutes Boateng?
Metadata: {{"numerical_metadata": {{"type": "jersey_number", "values": [12, 18, 2, 9]}}}}

Output:
{{
  "questions": [
    "Is Boateng visible in this event?",
    "Is a player substitution happening?",
    "What is the jersey number of the substituting player?"
  ]
}}

---

**Example 4: Discriminative Features Question**
Question: Which statement about a truck is true?
Metadata: {{"discriminative_features": {{"comparisons": [{{"dimension": "cab_color", "values": ["red", "blue", "white"]}}]}}}}

Output:
{{
  "questions": [
    "Is a truck visible in this event?",
    "What color is the truck's cab?",
    "What color is the truck's body?"
  ]
}}

---

**Example 5: Temporal Sequence Question**
Question: What does the assistant do after assembling the ice cream machine?
Metadata: {{"temporal_modifiers": {{"sequence_type": "after"}}, "discriminative_features": {{"comparisons": [{{"dimension": "action", "values": ["pour water", "open milk"]}}]}}}}

Output:
{{
  "questions": [
    "Is the shop assistant visible in this event?",
    "Is the ice cream machine visible?",
    "Is the assistant assembling the machine in this event?",
    "What action is the assistant performing?",
    "Is the assistant pouring water or opening milk?"
  ]
}}

---

**Example 6: Mixed Time (Localization + Content)**
Question: What's the time on the clock at 22:35?
Metadata: {{"content_times": {{"values": ["5:09", "4:09"], "context": "clock"}}}}

Output:
{{
  "questions": [
    "Is a clock visible in this event?",
    "What time is displayed on the clock?"
  ]
}}

---

**Example 7: Route Number**
Question: What was the bus route number?
Metadata: {{"numerical_metadata": {{"type": "route_number", "values": [9654, 8564, 9564, 8693]}}}}

Output:
{{
  "questions": [
    "Is a bus visible in this event?",
    "Is the route number visible on the bus?",
    "What is the route number displayed?"
  ]
}}

---

**Your Task:**
Question: {question}
Options: {options}
Metadata: {metadata}

Output (JSON only):
"""