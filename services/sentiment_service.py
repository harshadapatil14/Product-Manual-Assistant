"""
Sentiment Analysis Service for feedback processing
"""
import re
from typing import Dict, List, Tuple
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class SentimentService:
    """Service for analyzing sentiment of text feedback"""
    
    def __init__(self):
        self._download_nltk_data()
        self.sia = SentimentIntensityAnalyzer()
        
        # Extended word lists for better accuracy
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'awesome', 'perfect', 'love', 'like', 'helpful', 'useful', 'easy',
            'intuitive', 'fast', 'accurate', 'reliable', 'satisfied', 'happy',
            'brilliant', 'outstanding', 'superb', 'magnificent', 'delightful',
            'pleased', 'content', 'grateful', 'impressed', 'smooth', 'efficient',
            'convenient', 'user-friendly', 'responsive', 'quick', 'precise',
            'trustworthy', 'dependable', 'stable', 'robust', 'powerful'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrating',
            'difficult', 'confusing', 'slow', 'inaccurate', 'unreliable', 'hate',
            'dislike', 'annoying', 'broken', 'error', 'problem', 'issue',
            'useless', 'worthless', 'poor', 'mediocre', 'inadequate', 'inferior',
            'defective', 'faulty', 'buggy', 'crashed', 'failed', 'unresponsive',
            'laggy', 'clunky', 'awkward', 'complicated', 'overwhelming', 'stressful'
        }
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using multiple methods for better accuracy
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not text.strip():
            return self._empty_result()
        
        # Method 1: VADER Sentiment Analysis
        vader_scores = self.sia.polarity_scores(text)
        
        # Method 2: TextBlob Sentiment Analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Method 3: Custom word-based analysis
        custom_analysis = self._custom_word_analysis(text)
        
        # Combine results for final sentiment
        final_sentiment = self._combine_sentiment_scores(
            vader_scores, textblob_polarity, custom_analysis
        )
        
        return {
            "text": text,
            "final_sentiment": final_sentiment,
            "vader_scores": vader_scores,
            "textblob": {
                "polarity": textblob_polarity,
                "subjectivity": textblob_subjectivity
            },
            "custom_analysis": custom_analysis,
            "word_count": len(text.split()),
            "character_count": len(text)
        }
    
    def _custom_word_analysis(self, text: str) -> Dict:
        """Custom word-based sentiment analysis"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return {
                "score": 0,
                "positive_words": 0,
                "negative_words": 0,
                "total_words": 0
            }
        
        score = (positive_count - negative_count) / total_words
        
        return {
            "score": score,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "total_words": total_words
        }
    
    def _combine_sentiment_scores(self, vader_scores: Dict, textblob_polarity: float, 
                                 custom_analysis: Dict) -> Dict:
        """Combine multiple sentiment analysis methods"""
        
        # Weighted average of different methods
        vader_weight = 0.4
        textblob_weight = 0.3
        custom_weight = 0.3
        
        combined_score = (
            vader_scores['compound'] * vader_weight +
            textblob_polarity * textblob_weight +
            custom_analysis['score'] * custom_weight
        )
        
        # Determine sentiment category
        if combined_score > 0.1:
            sentiment = "Positive"
            emoji = "😊"
            color = "green"
        elif combined_score < -0.1:
            sentiment = "Negative"
            emoji = "😞"
            color = "red"
        else:
            sentiment = "Neutral"
            emoji = "😐"
            color = "orange"
        
        return {
            "sentiment": sentiment,
            "emoji": emoji,
            "color": color,
            "score": combined_score,
            "confidence": abs(combined_score)
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result for empty text"""
        return {
            "text": "",
            "final_sentiment": {
                "sentiment": "Neutral",
                "emoji": "😐",
                "color": "orange",
                "score": 0,
                "confidence": 0
            },
            "vader_scores": {"neg": 0, "neu": 1, "pos": 0, "compound": 0},
            "textblob": {"polarity": 0, "subjectivity": 0},
            "custom_analysis": {
                "score": 0,
                "positive_words": 0,
                "negative_words": 0,
                "total_words": 0
            },
            "word_count": 0,
            "character_count": 0
        }
    
    def get_sentiment_summary(self, analysis_result: Dict) -> str:
        """Get a human-readable summary of the sentiment analysis"""
        sentiment = analysis_result['final_sentiment']
        vader = analysis_result['vader_scores']
        custom = analysis_result['custom_analysis']
        
        summary = f"Sentiment: {sentiment['emoji']} {sentiment['sentiment']}\n"
        summary += f"Confidence: {sentiment['confidence']:.2f}\n"
        summary += f"Positive words: {custom['positive_words']}\n"
        summary += f"Negative words: {custom['negative_words']}\n"
        summary += f"Total words: {custom['total_words']}"
        
        return summary 