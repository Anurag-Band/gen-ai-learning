from dotenv import load_dotenv
from openai import OpenAI
import os
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  

# Initialize client
client = OpenAI(
    api_key=api_key,
    # base_url="https://generativelanguage.googleapis.com/v1beta/"
)


SYSTEM_PROMPT = """
You are an intelligent Ai assistant, expert in resolving users query using chain of thoughs COT method.
You work on START, PLAN & OUTPUT steps.
You need to first understand users query & then PLAN on what needs to be done, Then PLAN can be multiple steps.
Once you think enough plan has been done, then finally you can give OUTPUT

Rules: 
- Strictly follow given output format.
- Only run one step at a time.
- The sequence of the steps is START (where user gives an input), PLAN (that can be multiple times) & finally (which is going to displayed to user)

Output JSON format: 
{ "step": "START" | "PLAN" | "OUTPUT", content: "string" }

Example:
START: Hey, Can you solve 2 + 3 * 5 / 10
PLAN: { "step": "PLAN": "content": "Seems like user is interested in math problem" }
PLAN: { "step": "PLAN": "content": "looking at the problem, we should solve this using BODMAS method" }
PLAN: { "step": "PLAN": "content": "Yes, The BODMAS is correct thing to be done here" }
PLAN: { "step": "PLAN": "content": "first we must multiply 3 * 5 which is 15" }
PLAN: { "step": "PLAN": "content": "Now the new equation is 2 + 15 / 10" }
PLAN: { "step": "PLAN": "content": "We must perform divide that is 15 / 10 = 1.5" }
PLAN: { "step": "PLAN": "content": "Now the new equation is 2 + 1.5" }
PLAN: { "step": "PLAN": "content": "Now finally lets perform the add 3.5" }
PLAN: { "step": "PLAN": "content": "Great, we have solved and finally left with 3.5 as answer" }
OUTPUT: { "step": "OUTPUT": "content": "so, 2 + 3 * 5 / 10 = 3.5" }
"""

print("\n\n")

user_query = input("👉 ")

message_history = [
    { "role": "system", "content": SYSTEM_PROMPT }
]

message_history.append({ "role": "user", "content": user_query })

while True: 
    response = client.chat.completions.create(
    model="gpt-5-mini",
    response_format={"type": "json_object"},
    messages=message_history
)
    raw_result = response.choices[0].message.content
    
    message_history.append({ "role": "assistant", "content": raw_result })
    
    parced_result = json.loads(raw_result)
    
    if parced_result.get("step") == "START":
        print("💫", parced_result.get("content"))
        continue
    
    if parced_result.get("step") == "PLAN":
        print("🧠", parced_result.get("content"))
        continue
    
    if parced_result.get("step") == "OUTPUT":
        print("🏁", parced_result.get("content"))
        break


print("\n\n")