# WhatsApp Chat Analysis Tool

A comprehensive Python tool for analyzing WhatsApp chat exports with advanced NLP-based spam detection, data visualization, and statistical insights.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Output Explained](#output-explained)
- [Code Structure](#code-structure)

## 🎯 Overview

This tool takes a WhatsApp chat export file (`.txt` format) and performs comprehensive analysis including:
- Message parsing and data extraction
- Time-based activity analysis
- Word frequency analysis
- Visual data representations
- **Advanced NLP-based spam detection**

Perfect for analyzing group chats, understanding communication patterns, and identifying spam messages.

## ✨ Features

### 1. **Data Parsing & Cleaning**
- Extracts messages, timestamps, and user information from WhatsApp export
- Handles different message formats (user messages, system notifications)
- Cleans and standardizes date/time formats

### 2. **Time-Based Analysis**
- Extracts year, month, day, hour, and minute from each message
- Analyzes activity patterns by day and month
- Creates visualizations showing message distribution

### 3. **Data Visualization**
- **Bar Chart**: Messages sent per day
- **Heatmap**: Activity patterns across months and days
- **Word Cloud**: Visual representation of most frequently used words

### 4. **Word Frequency Analysis**
- Counts word occurrences
- Filters out common words like "media" and "omitted"
- Identifies top 5 most common words

### 5. **Advanced NLP Spam Detection** 🚀
- Uses **Machine Learning** (Naive Bayes classifier)
- **TF-IDF vectorization** for text analysis
- **10+ feature extraction** including:
  - Spam keyword detection
  - URL detection
  - Capitalization patterns
  - Punctuation analysis
  - Message length analysis
  - Media detection
- Provides **confidence scores** for each prediction
- Shows detailed indicators for why messages were flagged

## 📦 Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries
```
pandas          # Data manipulation and analysis
numpy           # Numerical computing
matplotlib      # Data visualization
seaborn         # Statistical data visualization
wordcloud       # Word cloud generation
scikit-learn    # Machine learning (NLP spam detection)
```

## 🔧 Installation

### Step 1: Install Python
Make sure Python is installed on your system. Download from [python.org](https://www.python.org/downloads/)

### Step 2: Install Required Packages
Open your terminal/command prompt and run:

```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn
```

Or install all at once:
```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn
```

## 📱 Usage

### Step 1: Export WhatsApp Chat
1. Open WhatsApp on your phone
2. Go to the chat/group you want to analyze
3. Tap the three dots (menu) → **More** → **Export chat**
4. Choose **Without Media** (to get a smaller file)
5. Save the file (it will be named something like `WhatsApp Chat with [Name].txt`)

### Step 2: Prepare the File
1. Rename the exported file to: `WhatsApp Chat with Section D.txt`
   - **OR** edit line 5 in the code to match your file name:
   ```python
   with open('YOUR_FILE_NAME.txt','r',encoding='utf-8') as f:
   ```

### Step 3: Run the Script
```bash
python whatsapp_chat_analysis.py
```

### Step 4: View Results
- Visualizations will open in separate windows
- Spam detection results will be printed in the terminal
- Summary statistics will be displayed at the end

## 🔍 How It Works

### Part 1: Data Loading & Parsing

```python
# Reads the WhatsApp export file
with open('WhatsApp Chat with Section D.txt','r',encoding='utf-8') as f:
    data = f.read()
```

**What happens:**
- Opens the chat export file
- Reads all content into memory
- Uses UTF-8 encoding to handle special characters

### Part 2: Message Extraction

```python
pattern = r'(\d{1,2}/\d{1,2}/\d{1,2},\s\d{1,2}:\d{2}\s[ap][m])'
messages = re.split(pattern,data)[1:]
```

**What happens:**
- Uses regex to find timestamps (e.g., "14/04/24, 3:09 pm")
- Splits the chat into individual messages
- Separates timestamps from message content
- Removes empty messages

### Part 3: User & Message Separation

```python
for msg in df['user_msg']:
    entry = re.split(r'([\w\W]+?):\s',msg)
    if entry[1:]:  # Has user name
        users.append(entry[1])
        message.append(entry[2])
    else:  # System notification
        users.append('group_notification')
        message.append(entry[0])
```

**What happens:**
- Extracts sender name from each message
- Separates user messages from system notifications
- Handles messages without sender names (group notifications)

### Part 4: Time Analysis

```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month_name()
df['day'] = df['date'].dt.day
df['hours'] = df['date'].dt.hour
```

**What happens:**
- Converts dates to datetime format
- Extracts time components (year, month, day, hour, minute)
- Enables time-based analysis and grouping

### Part 5: Visualizations

#### Bar Chart - Messages by Day
```python
hours_group = df.groupby('day').size()
hours_group.plot(kind='bar',color='blue')
```

**Shows:** How many messages were sent on each day of the month

#### Heatmap - Activity Patterns
```python
heatmap_data = df.groupby(['month','day']).size().unstack(fill_value=0)
sns.heatmap(heatmap_data, cmap='inferno')
```

**Shows:** Activity intensity across different months and days (darker = more active)

#### Word Cloud
```python
wordcloud = WordCloud(width=800, height=400).generate(txt)
plt.imshow(wordcloud)
```

**Shows:** Most frequently used words, with size indicating frequency

### Part 6: NLP Spam Detection (Advanced)

#### Step 1: Text Preprocessing
```python
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = ' '.join(text.split())  # Remove extra spaces
    return text
```

**Purpose:** Standardizes text for analysis

#### Step 2: Feature Extraction
The system extracts 10 different features from each message:

1. **Spam Keywords**: Counts suspicious words like "win", "free", "prize"
2. **URL Detection**: Checks for links (http, www, .com)
3. **Capitalization Ratio**: Detects excessive ALL CAPS
4. **Punctuation Ratio**: Identifies excessive punctuation
5. **Number Detection**: Finds phone numbers or numeric patterns
6. **Message Length**: Flags very short or very long messages
7. **Media Detection**: Identifies "<Media omitted>" messages
8. **Excessive Punctuation**: Flags multiple !!! or ???
9. **Upper Case Ratio**: Percentage of uppercase letters
10. **Message Length**: Total character count

#### Step 3: TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),  # Single words and word pairs
    stop_words='english'
)
X_tfidf = vectorizer.fit_transform(processed_messages)
```

**What is TF-IDF?**
- **TF (Term Frequency)**: How often a word appears in a message
- **IDF (Inverse Document Frequency)**: How rare/common a word is across all messages
- Converts text into numerical features that ML models can understand

#### Step 4: Machine Learning Classification
```python
clf = MultinomialNB(alpha=1.0)  # Naive Bayes classifier
clf.fit(X_combined, y_train)  # Train the model
predictions = clf.predict(X_combined)  # Make predictions
```

**How it works:**
- Combines TF-IDF features with handcrafted features
- Trains a Naive Bayes classifier (good for text classification)
- Predicts spam probability for each message
- Provides confidence scores (0-100%)

## 📊 Output Explained

### Visualization Outputs

1. **Bar Chart**: Shows message count per day
   - X-axis: Day of month (1-31)
   - Y-axis: Number of messages
   - Helps identify most active days

2. **Heatmap**: Shows activity patterns
   - X-axis: Day of month
   - Y-axis: Month name
   - Color intensity: Message count (darker = more messages)
   - Helps identify peak activity periods

3. **Word Cloud**: Visual word frequency
   - Larger words = more frequently used
   - Helps identify common topics/themes

### Spam Detection Output

```
=== Training NLP Spam Detection Model ===
Model Accuracy: 99.55%
Spam messages detected: 2 out of 220

[SPAM] (Confidence: 75.7%)
  Message: 'https://forms.gle/6Y5HgTphhZPU8wRv5'
  Indicators: URL, 1 spam keywords

=== Summary ===
Total messages analyzed: 220
Spam messages detected: 2 (0.9%)
Legitimate messages: 218 (99.1%)
```

**Understanding the Output:**
- **Model Accuracy**: How well the model performs (higher is better)
- **Confidence**: Probability that the message is spam (0-100%)
- **Indicators**: Why the message was flagged (URL, Media, keywords)
- **Summary**: Overall statistics

## 🏗️ Code Structure

```
whatsapp_chat_analysis.py
│
├── Imports & Setup (Lines 1-2)
│   └── Import required libraries
│
├── Data Loading (Lines 4-7)
│   └── Read WhatsApp export file
│
├── Message Parsing (Lines 12-30)
│   ├── Extract timestamps
│   ├── Split messages
│   └── Clean data
│
├── DataFrame Creation (Lines 32-72)
│   ├── Create pandas DataFrame
│   ├── Extract user names
│   ├── Parse dates
│   └── Extract time components
│
├── Visualizations (Lines 74-119)
│   ├── Bar chart (messages by day)
│   ├── Heatmap (activity patterns)
│   └── Word cloud (frequent words)
│
└── NLP Spam Detection (Lines 123-337)
    ├── Feature extraction functions
    ├── Text preprocessing
    ├── ML model training
    └── Spam prediction & reporting
```

## 🎓 Key Concepts Explained Simply

### What is NLP?
**Natural Language Processing (NLP)** = Teaching computers to understand human language
- In this code: Used to analyze message patterns and detect spam

### What is Machine Learning?
**Machine Learning (ML)** = Computer learns patterns from data
- In this code: The model learns what spam messages look like

### What is TF-IDF?
**Term Frequency-Inverse Document Frequency** = A way to convert words into numbers
- **Term Frequency**: How often a word appears
- **Inverse Document Frequency**: How unique/rare a word is
- **Result**: Numbers that represent word importance

### What is Naive Bayes?
**Naive Bayes Classifier** = A simple but effective ML algorithm
- Good for text classification (like spam detection)
- Fast and works well with text data
- Uses probability to make predictions

## 🔧 Customization

### Change Spam Keywords
Edit line 131-133:
```python
spam_keywords = ["win", "free", "click here", "prize", "urgent", ...]
```

### Adjust Spam Detection Sensitivity
Edit line 218 in `create_training_labels()`:
```python
labels.append(1 if score >= 3 else 0)  # Lower threshold = more sensitive
```

### Change Visualization Colors
Edit line 80:
```python
hours_group.plot(kind='bar', color='red')  # Change 'blue' to any color
```

## ⚠️ Common Issues & Solutions

### Issue: FileNotFoundError
**Problem**: Can't find the WhatsApp export file
**Solution**: 
- Make sure the file is in the same folder as the script
- Check the filename matches exactly (case-sensitive)
- Or edit line 5 to use your file name

### Issue: ModuleNotFoundError
**Problem**: Missing required library
**Solution**: Install missing package
```bash
pip install [package_name]
```

### Issue: No visualizations showing
**Problem**: Plots not displaying
**Solution**: 
- Make sure matplotlib backend is working
- On some systems, you may need: `plt.show(block=True)`

### Issue: Encoding errors
**Problem**: Special characters not displaying correctly
**Solution**: The code already uses UTF-8 encoding, but if issues persist, try:
```python
with open('file.txt', 'r', encoding='utf-8-sig') as f:
```

## 📈 Future Improvements

Potential enhancements you could add:
- Sentiment analysis (positive/negative messages)
- User activity ranking
- Message response time analysis
- Export results to CSV/Excel
- Interactive dashboard (using Plotly)
- Real-time chat monitoring
- Multi-language support
- Custom spam word lists per group

## 📝 Notes

- The spam detection uses **unsupervised learning** (creates labels from heuristics)
- For better accuracy, you could manually label some messages and use supervised learning
- The model trains on the same data it predicts (for demonstration)
- In production, use separate training and test sets

## 🤝 Contributing

Feel free to:
- Add new features
- Improve spam detection accuracy
- Add more visualizations
- Optimize code performance
- Fix bugs

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Uses scikit-learn for machine learning
- Uses pandas for data manipulation
- Uses matplotlib/seaborn for visualizations

---

**Made with ❤️ for WhatsApp chat analysis**

For questions or issues, please check the code comments or refer to the documentation of the libraries used.

