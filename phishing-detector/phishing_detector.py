# phishing_detector.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Dataset ---
emails = [
    ("URGENT: Verify your account now or it will be suspended. Click: http://secure-login.xyz/verify.php", "phishing"),
    ("You've won $1,000,000! Claim at http://prize-claim.net. Act NOW!", "phishing"),
    ("Your PayPal is limited. Login: http://192.168.1.1/paypal-secure.html", "phishing"),
    ("ACTION REQUIRED: Update payment at http://billing-update.xyz in 48 hours.", "phishing"),
    ("Suspicious login detected. Secure your account: http://bit.ly/3xKl99s", "phishing"),
    ("Claim your $500 Amazon Gift Card at http://amaz0n-gifts.com. Provide SSN.", "phishing"),
    ("Your bank account is frozen. Verify at http://chase-secure.net/unfreeze", "phishing"),
    ("IRS Tax Refund of $2,847. Claim at http://irs-refunds.xyz with your SSN.", "phishing"),
    ("Microsoft license expired. Renew at http://ms-office-renew.com now.", "phishing"),
    ("Package failed delivery. Pay $2.99 at http://fedx-delivery.net/reschedule", "phishing"),
    ("Netflix payment failed. Update at http://netflix-billing.xyz or cancelled.", "phishing"),
    ("Email storage full. Upgrade FREE at http://mail-upgrade.xyz NOW!", "phishing"),
    ("Inheritance $4.5M available. Contact lawyer2024@gmail.com to claim.", "phishing"),
    ("Dropbox files deleted in 24 hours. Save at http://dropbox-save.net", "phishing"),
    ("Apple ID suspended. Verify at http://apple-id-verify.com immediately.", "phishing"),

    ("Hi team, here are the standup notes. Alice is on the dashboard redesign.", "safe"),
    ("Hey Sarah, are you free for lunch today at 12:30 at the Thai place?", "safe"),
    ("Please find attached the Q3 marketing report. Let's discuss on Friday.", "safe"),
    ("Development is on track for the October 15th release. See Gantt chart.", "safe"),
    ("Welcome aboard Mark! Jennifer will meet you at reception at 9 AM Monday.", "safe"),
    ("Here's mom's lasagna recipe you asked for. Bake at 375F for 45 minutes.", "safe"),
    ("Conference Room B is booked for Thursday 2-4 PM. Projector available.", "safe"),
    ("Book club meeting is on the 22nd at 7 PM at Lisa's. We're reading Midnight Library.", "safe"),
    ("Invoice #2024-1089 of $3,450.00 has been received. Thank you.", "safe"),
    ("Hiking trip Saturday! Meet at trailhead at 8 AM. Bring water and snacks.", "safe"),
    ("Order #ORD-78234 shipped via UPS. Track at ups.com with 1Z999AA1012345678.", "safe"),
    ("Performance review scheduled Oct 20 at 3 PM. Prepare self-assessment by Oct 17.", "safe"),
    ("Dentist appointment with Dr. Patel on Friday Oct 15 at 10:30 AM.", "safe"),
    ("New espresso machine in the 3rd floor break room. Please clean after use.", "safe"),
    ("Neighborhood association meeting Oct 18 at 7 PM. Topics: lighting and park.", "safe"),
]

df = pd.DataFrame(emails, columns=["text", "label"])

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# --- Pipeline ---
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")),
    ("clf", MultinomialNB(alpha=0.1))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Results ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Confusion matrix plot ---
cm = confusion_matrix(y_test, y_pred, labels=["phishing", "safe"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Phishing", "Safe"],
            yticklabels=["Phishing", "Safe"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --- Try your own email ---
test_emails = [
    "URGENT: Your account will be deleted. Click http://verify-now.xyz immediately!",
    "Hi John, can we reschedule our meeting to 3 PM on Wednesday?",
]
predictions = model.predict(test_emails)
for email, pred in zip(test_emails, predictions):
    print(f"\n[{pred.upper()}] {email[:60]}...")