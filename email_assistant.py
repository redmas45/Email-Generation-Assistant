import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# Prompt Builder
# -----------------------------
def build_prompt(intent, facts, tone, word_limit):

    min_words = int(word_limit * 0.9)
    max_words = int(word_limit * 1.1)

    examples = """
Example 1:
Intent: Request leave
Facts: sick, 2 days leave, starting tomorrow
Tone: formal

Email:
Subject: Request for Sick Leave

Dear Manager,
I am writing to inform you that I am unwell and will require leave for the next two days starting tomorrow. I will ensure all pending work is managed accordingly.

Kind regards,
Rajiv Kumar

---

Example 2:
Intent: Apologize for delay
Facts: missed deadline, report submission, will send by evening
Tone: apologetic

Email:
Subject: Apology for Delay in Report Submission

Dear Team,
I sincerely apologize for missing the deadline for the report submission. I will ensure the report is completed and shared by this evening.

Best regards,
Rajiv Kumar

---

Example 3:
Intent: Request work from home
Facts: internet issue, 1 day, will be available online
Tone: formal

Email:
Subject: Request for Work From Home

Dear Manager,
Due to an internet issue at my location, I would like to request permission to work from home for one day. I will remain available online and ensure all tasks are completed.

Kind regards,
Rajiv Kumar

---

Example 4:
Intent: Follow up on payment
Facts: invoice #4567, pending payment, due last week
Tone: professional

Email:
Subject: Follow-Up on Pending Payment

Dear Sir/Madam,
I am writing to follow up on the pending payment for invoice #4567, which was due last week. Kindly process the payment at your earliest convenience.

Best regards,
Rajiv Kumar

---

Example 5:
Intent: Resign from job
Facts: last working day 30th April, personal reasons
Tone: formal

Email:
Subject: Resignation Notice

Dear Manager,
I would like to formally resign from my position due to personal reasons. My last working day will be 30th April.

Sincerely,
Rajiv Kumar

---

Example 6:
Intent: Invite to meeting
Facts: project discussion, tomorrow 3 PM, Zoom
Tone: professional

Email:
Subject: Meeting Invitation for Project Discussion

Dear Team,
You are invited to attend a meeting for project discussion scheduled tomorrow at 3 PM via Zoom.

Regards,
Rajiv Kumar

---

Example 7:
Intent: Complaint about service
Facts: delayed delivery, order #7890, unacceptable service
Tone: firm

Email:
Subject: Complaint Regarding Delayed Delivery

Dear Support Team,
I am writing to express my dissatisfaction with the delayed delivery of my order #7890. This level of service is unacceptable.

Regards,
Rajiv Kumar

---

Example 8:
Intent: Thank client
Facts: successful collaboration, project completion
Tone: appreciative

Email:
Subject: Thank You for Successful Collaboration

Dear Client,
Thank you for the successful collaboration and completion of the project. Your support and cooperation were greatly appreciated.

Best regards,
Rajiv Kumar

---

Example 9:
Intent: Request document
Facts: need report, urgent, by today evening
Tone: urgent

Email:
Subject: Urgent Request for Report

Dear Team,
I urgently need the report to be shared by today evening. Please prioritize this request.

Regards,
Rajiv Kumar

---

Example 10:
Intent: Congratulate colleague
Facts: promotion, well deserved
Tone: casual

Email:
Subject: Congratulations on Your Promotion

Hi,
Congratulations on your well-deserved promotion! Wishing you continued success.

Cheers,
Rajiv Kumar

---

Example 11:
Intent: Reminder for meeting
Facts: meeting today 5 PM, don't forget
Tone: friendly

Email:
Subject: Friendly Reminder for Meeting

Hi Team,
Just a reminder about the meeting scheduled today at 5 PM. Looking forward to your participation.

Thanks,
Rajiv Kumar

---

Example 12:
Intent: Apologize to manager
Facts: missed call, was in meeting, will call back
Tone: apologetic

Email:
Subject: Apology for Missed Call

Dear Manager,
I apologize for missing your call earlier as I was in a meeting. I will call you back shortly.

Sincerely,
Rajiv Kumar
---
"""

    prompt = f"""
You are a senior corporate email writing assistant.

### TASK:
Generate a high-quality email.

### STRICT RULES:
- Include Subject line
- Maintain tone: {tone}
- MUST include ALL facts: {facts}
- Keep word count BETWEEN {min_words} and {max_words}
- Structure: Subject → Greeting → Body → Closing

### SELF-CHECK:
Before finalizing:
- Ensure all facts are included
- Ensure tone matches exactly
- Ensure word count is within range

{examples}

### INPUT:
Intent: {intent}
Facts: {facts}
Tone: {tone}
Word Limit: {word_limit}

### OUTPUT:
Email:
"""

    return prompt


# -----------------------------
# Email Generator
# -----------------------------
def generate_email(intent, facts, tone, word_limit, model="llama-3.3-70b-versatile"):
    prompt = build_prompt(intent, facts, tone, word_limit)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()


# -----------------------------
# Retry Mechanism
# -----------------------------
def generate_email_with_retry(intent, facts, tone, word_limit, retries=2):
    for _ in range(retries):
        email = generate_email(intent, facts, tone, word_limit)
        wc = len(email.split())

        if word_limit * 0.9 <= wc <= word_limit * 1.1:
            return email

    return email


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    intent = input("Enter intent: ")
    facts = input("Enter key facts (comma-separated): ")
    tone = input("Enter tone: ")
    word_limit = int(input("Enter word limit (e.g., 100): "))

    email = generate_email_with_retry(intent, facts, tone, word_limit)

    print("\nGenerated Email:\n")
    print(email)

    # Word count
    word_count = len(email.split())
    #print(f"\n📊 Word Count: {word_count}")

    # Save to file
    with open("generated_email.txt", "w", encoding="utf-8") as f:
        f.write(email)

    print("\n✅ Email saved to generated_email.txt")