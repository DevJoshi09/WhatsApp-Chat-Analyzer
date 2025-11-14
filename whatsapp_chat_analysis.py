import re as re
import pandas as pd

# Data loaded into memory  
with open('WhatsApp Chat with Section D.txt','r',encoding='utf-8') as f:
    # The data is stored as a string 
    data = f.read()

# The Raw data is printed for verification
# print(data)

# Parsing messages - identifies timestamps in chat data.
pattern = r'(\d{1,2}/\d{1,2}/\d{1,2},\s\d{1,2}:\d{2}\s[ap][m])'

# spliting the chat content into seprate message
messages = re.split(pattern,data)[1:]
# remove the timestamps from message
msg_cleaned = [re.sub(pattern,'', msg) for msg in messages]
# filtering witespace or empty message
cleaned_messages = list(filter(lambda x: x.strip() != '',msg_cleaned ))

cleaned_messages


# date corresponding to each message are extracted
dates = re.findall(pattern,data)
# Remove all \u202f characters
cleaned_dates = [date.replace('\u202f', ' ') for date in dates]
cleaned_dates


from datetime import datetime

# creating dataframe to store message and corresponding dates
df = pd.DataFrame({'user_msg':cleaned_messages,'date':cleaned_dates})
# dates are converted to standarized datetime format
df['date'] = pd.to_datetime(df['date'],format='%d/%m/%y, %I:%M %p')
df.head()

#seprate user name and message
users = []
message =[]
for msg in df['user_msg']:
    entry = re.split(r'([\w\W]+?):\s',msg)
    if entry[1:]:# user name
        users.append(entry[1])
        message.append(entry[2])
    else:
        # message without label
        users.append('group_notification')
        message.append(entry[0])

df['user']=users
df['messages']= message
df.drop(columns=['user_msg'],inplace=True,errors='ignore')
df.head()


df['year']=df['date'].dt.year  #dt is use for using datetime like propertiese of pandas

df['month']=df['date'].dt.month_name()

df['day']=df['date'].dt.day

df['hours']=df['date'].dt.hour

df['minute']=df['date'].dt.minute

df.head()

df.drop(columns=['date'],inplace=True,errors='ignore')
df.head()

#visualization

import matplotlib.pyplot as plt

hours_group = df.groupby('day').size()
plt.figure(figsize=(10,5))
hours_group.plot(kind='bar',color='blue')
plt.title('Message Sent By Day')
plt.xlabel('Day')
plt.ylabel('Messages')
plt.show()

import seaborn as sns

#heatmap of day and hours in which users were most active
heatmap_data=df.groupby(['month','day']).size().unstack(fill_value=0)
sns.heatmap(heatmap_data,cmap='inferno')
plt.title('Heatmap of Activity')
plt.xlabel('Day')
plt.ylabel('Month')
plt.show()


from collections import Counter
#join all the  message and split into words
text = ' '.join(df['messages'])

#remove puntuation and split into words
words = re.findall(r'\w+',text)
words_to_remove={"media","omitted"}
filtered_words = [word for word in words if word.lower() not in words_to_remove]

word_count=Counter(filtered_words)
#print 5 most common words
most_common_words = word_count.most_common(5)
most_common_words

# Generate wordcloud
from wordcloud import WordCloud
txt=' '.join(filtered_words)
# used for frequently used words
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(txt)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('on')
plt.show()

df.columns

# ==================== NLP-Based Spam Detection ====================
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string

# Predefined spam indicators
spam_keywords = ["win", "free", "click here", "prize", "urgent", "limited time", 
                 "act now", "congratulations", "winner", "claim", "offer", 
                 "discount", "buy now", "click", "link", "http", "www"]

def preprocess_text(text):
    """Preprocess text for NLP analysis"""
    if pd.isna(text) or text == '':
        return ''
    # Convert to lowercase
    text = str(text).lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def extract_features(messages):
    """Extract various features from messages"""
    features = []
    for msg in messages:
        msg_str = str(msg).lower()
        msg_len = len(msg_str)
        
        # Feature 1: Contains spam keywords
        spam_keyword_count = sum(1 for keyword in spam_keywords if keyword in msg_str)
        
        # Feature 2: Contains URLs
        has_url = 1 if ('http' in msg_str or 'www.' in msg_str or '.com' in msg_str) else 0
        
        # Feature 3: Excessive capitalization (more than 30% uppercase)
        if msg_len > 0:
            upper_ratio = sum(1 for c in str(msg) if c.isupper()) / msg_len
        else:
            upper_ratio = 0
        
        # Feature 4: Excessive punctuation
        punct_count = sum(1 for c in str(msg) if c in string.punctuation)
        punct_ratio = punct_count / msg_len if msg_len > 0 else 0
        
        # Feature 5: Contains numbers (phone numbers, etc.)
        has_numbers = 1 if any(char.isdigit() for char in str(msg)) else 0
        
        # Feature 6: Message length (very short or very long might be spam)
        is_short = 1 if msg_len < 10 else 0
        is_very_long = 1 if msg_len > 500 else 0
        
        # Feature 7: Contains media omitted (common in WhatsApp)
        has_media = 1 if 'media omitted' in msg_str or '<media omitted>' in msg_str else 0
        
        # Feature 8: Excessive exclamation/question marks
        exclamation_count = str(msg).count('!')
        question_count = str(msg).count('?')
        excessive_punct = 1 if (exclamation_count > 3 or question_count > 3) else 0
        
        features.append({
            'spam_keywords': spam_keyword_count,
            'has_url': has_url,
            'upper_ratio': upper_ratio,
            'punct_ratio': punct_ratio,
            'has_numbers': has_numbers,
            'is_short': is_short,
            'is_very_long': is_very_long,
            'has_media': has_media,
            'excessive_punct': excessive_punct,
            'message_length': msg_len
        })
    return pd.DataFrame(features)

def create_training_labels(messages, features_df):
    """Create training labels based on heuristics (since we don't have labeled data)"""
    labels = []
    for idx, msg in enumerate(messages):
        msg_str = str(msg).lower()
        score = 0
        
        # Rule-based scoring
        if features_df.iloc[idx]['has_media']:
            score += 2
        if features_df.iloc[idx]['has_url']:
            score += 3
        if features_df.iloc[idx]['spam_keywords'] > 0:
            score += features_df.iloc[idx]['spam_keywords'] * 2
        if features_df.iloc[idx]['excessive_punct']:
            score += 2
        if features_df.iloc[idx]['upper_ratio'] > 0.3:
            score += 2
        if features_df.iloc[idx]['is_short'] and features_df.iloc[idx]['has_url']:
            score += 2
        
        # Label as spam if score >= 3
        labels.append(1 if score >= 3 else 0)
    return np.array(labels)

def detect_spam_nlp(messages):
    """Advanced NLP-based spam detection"""
    print("\n=== Training NLP Spam Detection Model ===")
    
    # Preprocess messages
    processed_messages = [preprocess_text(msg) for msg in messages]
    
    # Extract features
    features_df = extract_features(messages)
    
    # Create training labels using heuristics
    y_train = create_training_labels(messages, features_df)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    try:
        X_tfidf = vectorizer.fit_transform(processed_messages)
        
        # Combine TF-IDF features with handcrafted features
        X_combined = np.hstack([
            X_tfidf.toarray(),
            features_df.values
        ])
        
        # Train Naive Bayes classifier
        clf = MultinomialNB(alpha=1.0)
        clf.fit(X_combined, y_train)
        
        # Predict on the same data (in production, use separate test set)
        predictions = clf.predict(X_combined)
        probabilities = clf.predict_proba(X_combined)[:, 1]  # Probability of being spam
        
        # Calculate accuracy
        accuracy = accuracy_score(y_train, predictions)
        print(f"Model Accuracy: {accuracy:.2%}")
        print(f"Spam messages detected: {sum(predictions)} out of {len(messages)}")
        
        # Return results with confidence scores
        results = []
        for i, msg in enumerate(messages):
            results.append({
                'message': msg,
                'is_spam': bool(predictions[i]),
                'confidence': probabilities[i],
                'spam_score': int(features_df.iloc[i]['spam_keywords']),
                'has_url': bool(features_df.iloc[i]['has_url']),
                'has_media': bool(features_df.iloc[i]['has_media'])
            })
        
        return results
        
    except Exception as e:
        print(f"Error in NLP model: {e}")
        print("Falling back to rule-based detection...")
        # Fallback to rule-based
        results = []
        features_df = extract_features(messages)
        for i, msg in enumerate(messages):
            score = 0
            if features_df.iloc[i]['has_media']:
                score += 2
            if features_df.iloc[i]['has_url']:
                score += 3
            if features_df.iloc[i]['spam_keywords'] > 0:
                score += features_df.iloc[i]['spam_keywords'] * 2
            
            is_spam = score >= 3
            results.append({
                'message': msg,
                'is_spam': is_spam,
                'confidence': min(score / 10.0, 1.0),
                'spam_score': int(features_df.iloc[i]['spam_keywords']),
                'has_url': bool(features_df.iloc[i]['has_url']),
                'has_media': bool(features_df.iloc[i]['has_media'])
            })
        return results

# Detect spam using NLP
print("\n" + "="*60)
print("NLP-BASED SPAM DETECTION")
print("="*60)
results = detect_spam_nlp(df['messages'])

# Output the results with confidence scores
print("\n=== Spam Detection Results ===")
spam_count = 0
for result in results:
    status = "SPAM" if result['is_spam'] else "NOT SPAM"
    confidence = result['confidence']
    msg_preview = result['message'][:50] + "..." if len(result['message']) > 50 else result['message']
    
    if result['is_spam']:
        spam_count += 1
        indicators = []
        if result['has_url']:
            indicators.append("URL")
        if result['has_media']:
            indicators.append("Media")
        if result['spam_score'] > 0:
            indicators.append(f"{result['spam_score']} spam keywords")
        
        print(f"\n[{status}] (Confidence: {confidence:.1%})")
        print(f"  Message: '{msg_preview}'")
        print(f"  Indicators: {', '.join(indicators) if indicators else 'Pattern-based detection'}")

print(f"\n=== Summary ===")
print(f"Total messages analyzed: {len(results)}")
print(f"Spam messages detected: {spam_count} ({spam_count/len(results)*100:.1f}%)")
print(f"Legitimate messages: {len(results) - spam_count} ({(len(results)-spam_count)/len(results)*100:.1f}%)")

# THE END 
# THANKYOU

