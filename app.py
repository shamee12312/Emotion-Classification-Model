import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from emotion_classifier import EmotionClassifier
from data_processor import DataProcessor
from model_evaluator import ModelEvaluator
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Emotion Classification System",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

def load_sample_data():
    """Create a comprehensive, diverse emotion dataset for accurate training"""
    emotions = ['Happy', 'Sad', 'Angry', 'Neutral', 'Fear', 'Surprise']
    
    # Comprehensive sample texts for each emotion - extensive and realistic
    sample_texts = {
        'Happy': [
            "I'm so excited about this amazing opportunity!",
            "What a beautiful sunny day, feeling fantastic!",
            "Just got the best news ever, I'm thrilled!",
            "Love spending time with my family and friends",
            "This is the best day of my life!",
            "I feel so grateful and blessed today",
            "Amazing performance, I'm so proud!",
            "Everything is going perfectly, I'm delighted!",
            "Celebrating this wonderful achievement with joy!",
            "Feeling incredibly happy and content right now",
            "What a fantastic surprise, I'm overjoyed!",
            "Life is beautiful and I'm loving every moment",
            "Got promoted today, feeling on top of the world!",
            "Just had the most wonderful vacation ever",
            "My dreams are finally coming true, so happy!",
            "Spending quality time with loved ones brings me joy",
            "Successfully completed my project, feeling great!",
            "This positive energy is exactly what I needed",
            "Laughing so much my cheeks hurt, pure happiness",
            "Found the perfect solution, I'm thrilled beyond words",
            "Wedding day was absolutely magical and perfect",
            "My baby took their first steps today!",
            "Just received my acceptance letter to university",
            "Won the lottery, I can't contain my excitement!",
            "Reunited with my best friend after years",
            "The sunset tonight is breathtakingly beautiful",
            "My favorite team won the championship!",
            "Just finished reading an inspiring book",
            "Received an unexpected compliment that made my day",
            "Dancing all night at this incredible party",
            "My garden is blooming beautifully this spring",
            "Just bought my dream house, I'm ecstatic!",
            "Had the most delicious meal at a new restaurant",
            "My children graduated with honors today",
            "Feeling blessed to have such wonderful friends",
            "Just learned I'm going to be a grandparent!",
            "The music concert was absolutely phenomenal",
            "Achieved my fitness goals after months of hard work",
            "Waking up refreshed after a perfect night's sleep",
            "Found my lost cat safe and sound!"
        ],
        'Sad': [
            "I'm feeling really down today",
            "This is such a disappointing situation",
            "I miss my old friends so much",
            "Feeling lonely and isolated lately",
            "Nothing seems to be going right",
            "I'm having a really tough time",
            "This news made me feel so heartbroken",
            "I feel empty and lost right now",
            "Tears are flowing down my face uncontrollably",
            "Everything feels hopeless and dark today",
            "Lost my job and don't know what to do",
            "My pet passed away, I'm devastated",
            "Relationship ended badly, feeling broken inside",
            "Failed the exam despite studying so hard",
            "Nobody understands what I'm going through",
            "Feeling overwhelmed by all these problems",
            "Can't stop crying, everything hurts so much",
            "Disappointed by people I trusted most",
            "This rejection letter crushed my spirits",
            "Weather is gloomy and so is my mood",
            "Lost my grandmother last week, still grieving",
            "My best friend moved away to another country",
            "Diagnosed with a serious illness yesterday",
            "My dreams have been shattered completely",
            "Feeling worthless and unwanted by everyone",
            "The divorce papers arrived today, I'm devastated",
            "My childhood home is being demolished",
            "Can't afford to pay the bills this month",
            "Nobody showed up to my birthday party",
            "My artwork was rejected from the gallery",
            "Feeling like a failure in everything I do",
            "Lost all my savings in a bad investment",
            "My mentor passed away unexpectedly",
            "The surgery didn't go as planned",
            "Watching my parents age makes me so sad",
            "My favorite restaurant closed down permanently",
            "Feeling disconnected from everyone around me",
            "The rainy weather matches my melancholy mood",
            "My plants are dying despite my best efforts",
            "Remembering happier times makes me cry"
        ],
        'Angry': [
            "This is absolutely infuriating!",
            "I can't believe how unfair this is",
            "I'm so frustrated with this situation",
            "This makes my blood boil!",
            "I'm outraged by this behavior",
            "This is completely unacceptable!",
            "I'm fed up with all these problems",
            "This injustice makes me furious!",
            "How dare they treat me like this!",
            "This disrespect is making me livid",
            "I've had enough of these stupid rules",
            "Their incompetence is driving me crazy",
            "This traffic jam is absolutely maddening",
            "Can't believe they lied to my face",
            "This corrupt system makes me sick",
            "I'm so angry I can barely think straight",
            "They crossed the line this time, I'm furious",
            "This betrayal has made me absolutely livid",
            "Their arrogance is beyond infuriating",
            "I'm boiling with rage over this decision",
            "The customer service is absolutely terrible",
            "They keep ignoring my legitimate complaints",
            "This discrimination is completely unacceptable",
            "I'm furious about the broken promises",
            "Their rude behavior is absolutely disgusting",
            "This blatant disregard for rules infuriates me",
            "I'm sick of being treated like garbage",
            "This unfair treatment makes me see red",
            "They're taking advantage of innocent people",
            "This deliberate sabotage is infuriating",
            "I'm outraged by their complete lack of empathy",
            "This bureaucratic nonsense is driving me insane",
            "Their condescending attitude is unbearable",
            "I'm furious about the wasted time and money",
            "This dishonesty is absolutely revolting",
            "They're being deliberately obstructive and hostile",
            "This invasion of privacy makes me livid",
            "I'm angry about the environmental destruction",
            "Their hypocrisy is absolutely maddening",
            "This manipulation and gaslighting is unforgivable"
        ],
        'Neutral': [
            "The weather today is partly cloudy",
            "I need to go to the grocery store",
            "The meeting is scheduled for 3 PM",
            "Please review the attached document",
            "The report is due next Friday",
            "I'll be working from home tomorrow",
            "The conference will be held virtually",
            "Please confirm your attendance",
            "The office hours are 9 AM to 5 PM",
            "The project deadline is next month",
            "Please submit your application by Monday",
            "The system will undergo maintenance tonight",
            "Traffic conditions are normal on Highway 101",
            "The store closes at 8 PM on weekdays",
            "Meeting room A is available for booking",
            "Please update your contact information",
            "The quarterly review will be conducted soon",
            "Standard operating procedures must be followed",
            "The temperature today reached 72 degrees",
            "Registration opens at 9 AM sharp",
            "The documentation has been updated accordingly",
            "Please proceed to the next checkpoint",
            "The inventory count is scheduled for Thursday",
            "All systems are operating within normal parameters",
            "The training session will cover basic protocols",
            "Please refer to section 4.2 of the manual",
            "The backup process completed successfully",
            "New policy takes effect starting next quarter",
            "The facility will be closed for renovations",
            "Please complete the required safety training",
            "The audit will begin on the first of next month",
            "Standard warranty terms apply to this product",
            "The software update is currently being installed",
            "Please verify your identity before proceeding",
            "The shipment is expected to arrive Tuesday",
            "All participants must sign the waiver form",
            "The database migration is scheduled for tonight",
            "Please follow the established checkout procedure",
            "The equipment calibration is due next week",
            "Standard business hours apply during holidays"
        ],
        'Fear': [
            "I'm terrified of what might happen",
            "This situation makes me very anxious",
            "I'm worried about the future",
            "I feel scared and uncertain",
            "This gives me chills down my spine",
            "I'm afraid things will get worse",
            "The thought of this terrifies me",
            "I'm nervous about the outcome",
            "This makes me feel vulnerable and helpless",
            "I'm trembling with fear right now",
            "What if everything goes wrong tomorrow?",
            "The darkness scares me so much",
            "I'm afraid I'll fail this important test",
            "This medical diagnosis has me terrified",
            "Flying in airplanes makes me panic",
            "I'm scared of speaking in public",
            "The thought of losing my job frightens me",
            "I'm anxious about meeting new people",
            "This financial crisis has me worried sick",
            "I'm afraid of being alone in the dark",
            "The thought of death keeps me awake at night",
            "I'm terrified of heights and elevators",
            "This storm is making me very anxious",
            "I'm scared of what the test results might show",
            "The strange noises upstairs frighten me",
            "I'm afraid of losing my loved ones",
            "This uncertainty about the future terrifies me",
            "I'm nervous about the job interview tomorrow",
            "The thought of failure paralyzes me with fear",
            "I'm scared of making the wrong decision",
            "This economic instability has me worried",
            "I'm afraid of spiders and insects",
            "The thought of surgery makes me panic",
            "I'm terrified of being judged by others",
            "This global situation fills me with dread",
            "I'm scared of running out of money",
            "The thought of aging frightens me deeply",
            "I'm afraid of technology taking over",
            "This climate change news is terrifying",
            "I'm nervous about my children's safety"
        ],
        'Surprise': [
            "I can't believe this actually happened!",
            "What a shocking turn of events!",
            "I never expected this to occur",
            "This is completely unexpected!",
            "I'm amazed by this revelation",
            "What a surprising discovery!",
            "This caught me completely off guard",
            "I'm stunned by this news!",
            "Wow, this is absolutely incredible!",
            "I'm blown away by this surprise",
            "Never saw that plot twist coming!",
            "This outcome is beyond my wildest dreams",
            "What an unexpected gift, I'm speechless!",
            "I'm shocked by how well this turned out",
            "This sudden change left me speechless",
            "Incredible, I never thought this was possible",
            "What a remarkable coincidence this is!",
            "This surprise party blew my mind completely",
            "I'm astonished by this amazing discovery",
            "Such an unexpected but welcome surprise!",
            "I never imagined I would win this contest",
            "What a twist! Nobody predicted this ending",
            "I'm amazed by the sudden improvement",
            "This unexpected visitor made my day",
            "What a shocking revelation about my family history",
            "I'm stunned by this incredible transformation",
            "This sudden opportunity came out of nowhere",
            "What an amazing coincidence meeting you here!",
            "I'm surprised by how much I enjoyed this",
            "This unexpected compliment caught me off guard",
            "What a remarkable turn of events this is",
            "I'm amazed by this technological breakthrough",
            "This surprise announcement changed everything",
            "What an incredible stroke of luck!",
            "I'm shocked by this generous donation",
            "This unexpected solution is brilliant",
            "What a surprising reaction from the audience",
            "I'm astonished by this scientific discovery",
            "This sudden weather change is remarkable",
            "What an unexpected but delightful surprise!"
        ]
    }
    
    # Create DataFrame
    data = []
    for emotion, texts in sample_texts.items():
        for text in texts:
            data.append({'text': text, 'emotion': emotion})
    
    return pd.DataFrame(data)

def main():
    st.title("🎭 Emotion Classification System")
    st.markdown("**Complete ML Pipeline**: Data exploration → Model training → Evaluation → Deployment")
    
    # ML Pipeline Overview
    with st.expander("📊 ML Pipeline Overview", expanded=False):
        st.markdown("""
        ### End-to-End Machine Learning Workflow
        
        **1. Data Exploration & Analysis (EDA)**
        - Dataset statistics and emotion distribution
        - Text length analysis and feature exploration
        - Data quality assessment and preprocessing
        
        **2. Feature Engineering & Preprocessing**
        - Text cleaning and normalization
        - TF-IDF vectorization with n-grams
        - Tokenization and lemmatization
        
        **3. Model Training & Selection**
        - Multiple algorithm comparison (Naive Bayes, SVM, Random Forest, Logistic Regression)
        - Hyperparameter optimization
        - Cross-validation and stratified sampling
        
        **4. Model Evaluation & Metrics**
        - Accuracy, Precision, Recall, F1-Score
        - Confusion matrix analysis
        - Per-class performance assessment
        
        **5. Model Deployment & Inference**
        - Real-time prediction interface
        - Confidence score analysis
        - Batch prediction capabilities
        """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Overview", "Model Training", "Model Evaluation", "Prediction Interface"]
    )
    
    if page == "Data Overview":
        show_data_overview()
    elif page == "Model Training":
        show_model_training()
    elif page == "Model Evaluation":
        show_model_evaluation()
    elif page == "Prediction Interface":
        show_prediction_interface()

def show_data_overview():
    st.header("📊 Data Overview")
    
    # Load data
    st.subheader("Dataset")
    data_option = st.radio(
        "Choose data source:",
        ["Use Sample Data", "Upload Custom Data"]
    )
    
    if data_option == "Use Sample Data":
        df = load_sample_data()
        st.success("Sample emotion dataset loaded successfully!")
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV file with 'text' and 'emotion' columns",
            type=['csv']
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns or 'emotion' not in df.columns:
                    st.error("CSV file must contain 'text' and 'emotion' columns")
                    return
                st.success("Custom dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        else:
            st.info("Please upload a CSV file to continue")
            return
    
    # Store data in session state
    st.session_state.data = df
    
    # Display basic statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(df))
    
    with col2:
        unique_emotions = df['emotion'].nunique()
        st.metric("Unique Emotions", unique_emotions)
    
    with col3:
        avg_length = df['text'].str.len().mean()
        st.metric("Average Text Length", f"{avg_length:.1f}")
    
    # Show data sample
    st.subheader("Data Sample")
    st.dataframe(df.head(10))
    
    # Emotion distribution
    st.subheader("Emotion Distribution")
    emotion_counts = df['emotion'].value_counts()
    
    fig = px.bar(
        x=emotion_counts.index,
        y=emotion_counts.values,
        title="Distribution of Emotions in Dataset",
        labels={'x': 'Emotion', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Text length analysis
    st.subheader("Text Length Analysis")
    df['text_length'] = df['text'].str.len()
    
    fig = px.box(
        df,
        x='emotion',
        y='text_length',
        title="Text Length Distribution by Emotion"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_model_training():
    st.header("🤖 Model Training")
    
    if 'data' not in st.session_state:
        st.warning("Please load data first from the Data Overview page")
        return
    
    df = st.session_state.data
    
    # Training parameters
    st.subheader("Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.25, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=0)
    
    with col2:
        max_features = st.selectbox("Max Features (TF-IDF)", [2000, 5000, 8000], index=1)
        ngram_range = st.selectbox("N-gram Range", ["(1,1)", "(1,2)", "(1,3)"], index=1)
    
    # Model selection
    st.subheader("Model Selection")
    models_to_train = st.multiselect(
        "Select models to train:",
        ["Naive Bayes", "SVM", "Random Forest", "Logistic Regression"],
        default=["Naive Bayes", "SVM", "Random Forest"]
    )
    
    if st.button("Train Models", type="primary"):
        if not models_to_train:
            st.error("Please select at least one model to train")
            return
        
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Initialize classifier
                classifier = EmotionClassifier()
                
                # Parse ngram_range
                ngram_tuple = eval(ngram_range)
                
                # Train models
                results = classifier.train_models(
                    df,
                    test_size=test_size,
                    random_state=random_state,
                    max_features=max_features,
                    ngram_range=ngram_tuple,
                    models_to_train=models_to_train
                )
                
                # Store results in session state
                st.session_state.classifier = classifier
                st.session_state.models_trained = True
                st.session_state.training_results = results
                
                st.success("Models trained successfully!")
                
                # Display training results
                st.subheader("Training Results")
                
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df)
                
                # Best model
                best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
                st.success(f"Best performing model: **{best_model}** (Accuracy: {results[best_model]['accuracy']:.4f})")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

def show_model_evaluation():
    st.header("📈 Model Evaluation")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the Model Training page")
        return
    
    classifier = st.session_state.classifier
    results = st.session_state.training_results
    
    # Model selection for detailed evaluation
    selected_model = st.selectbox(
        "Select model for detailed evaluation:",
        list(results.keys())
    )
    
    if st.button("Generate Detailed Evaluation"):
        with st.spinner("Generating evaluation metrics..."):
            try:
                evaluator = ModelEvaluator(classifier)
                evaluation_results = evaluator.evaluate_model(selected_model)
                st.session_state.evaluation_results = evaluation_results
                
                # Display metrics with clearer formatting
                st.subheader("📊 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                accuracy_pct = evaluation_results['accuracy'] * 100
                precision_pct = evaluation_results['precision'] * 100
                recall_pct = evaluation_results['recall'] * 100
                f1_pct = evaluation_results['f1_score'] * 100
                
                with col1:
                    st.metric("Accuracy", f"{accuracy_pct:.1f}%", f"{evaluation_results['accuracy']:.4f}")
                
                with col2:
                    st.metric("Precision", f"{precision_pct:.1f}%", f"{evaluation_results['precision']:.4f}")
                
                with col3:
                    st.metric("Recall", f"{recall_pct:.1f}%", f"{evaluation_results['recall']:.4f}")
                
                with col4:
                    st.metric("F1-Score", f"{f1_pct:.1f}%", f"{evaluation_results['f1_score']:.4f}")
                
                # Performance indicator
                if accuracy_pct >= 90:
                    st.success(f"🎯 Excellent performance! {selected_model} achieved {accuracy_pct:.1f}% accuracy")
                elif accuracy_pct >= 80:
                    st.info(f"✅ Good performance! {selected_model} achieved {accuracy_pct:.1f}% accuracy")
                elif accuracy_pct >= 70:
                    st.warning(f"⚠️ Moderate performance. {selected_model} achieved {accuracy_pct:.1f}% accuracy")
                else:
                    st.error(f"❌ Low performance. {selected_model} achieved {accuracy_pct:.1f}% accuracy")
                
                # Enhanced Confusion Matrix
                st.subheader("🎯 Confusion Matrix Analysis")
                
                # Calculate normalized confusion matrix for better visualization
                cm = evaluation_results['confusion_matrix']
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Raw Counts**")
                    fig1 = px.imshow(
                        cm,
                        x=evaluation_results['labels'],
                        y=evaluation_results['labels'],
                        color_continuous_scale='Blues',
                        aspect='auto',
                        title=f"Confusion Matrix - {selected_model} (Counts)",
                        text_auto=True
                    )
                    fig1.update_layout(
                        xaxis_title="Predicted Emotion",
                        yaxis_title="Actual Emotion",
                        height=400
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.write("**Normalized Percentages**")
                    fig2 = px.imshow(
                        cm_normalized,
                        x=evaluation_results['labels'],
                        y=evaluation_results['labels'],
                        color_continuous_scale='RdYlBu_r',
                        aspect='auto',
                        title=f"Normalized Confusion Matrix - {selected_model}",
                        text_auto='.2%'
                    )
                    fig2.update_layout(
                        xaxis_title="Predicted Emotion",
                        yaxis_title="Actual Emotion",
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Confusion Matrix Insights
                st.subheader("📈 Model Insights")
                
                # Calculate per-class accuracy
                class_accuracies = {}
                for i, label in enumerate(evaluation_results['labels']):
                    class_total = cm[i].sum()
                    class_correct = cm[i][i]
                    class_accuracies[label] = (class_correct / class_total) if class_total > 0 else 0
                
                # Find best and worst performing emotions
                best_emotion = max(class_accuracies, key=class_accuracies.get)
                worst_emotion = min(class_accuracies, key=class_accuracies.get)
                
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                
                with insight_col1:
                    st.metric(
                        "Best Classified Emotion", 
                        best_emotion, 
                        f"{class_accuracies[best_emotion]:.1%}"
                    )
                
                with insight_col2:
                    st.metric(
                        "Most Challenging Emotion", 
                        worst_emotion, 
                        f"{class_accuracies[worst_emotion]:.1%}"
                    )
                
                with insight_col3:
                    avg_class_acc = np.mean(list(class_accuracies.values()))
                    st.metric(
                        "Average Class Accuracy", 
                        f"{avg_class_acc:.1%}",
                        f"Balanced: {'Yes' if abs(avg_class_acc - accuracy_pct/100) < 0.05 else 'No'}"
                    )
                
                # Classification Report
                st.subheader("Classification Report")
                report_df = pd.DataFrame(evaluation_results['classification_report'])
                st.dataframe(report_df)
                
                # Enhanced Model Comparison
                st.subheader("🔄 Model Comparison Dashboard")
                
                comparison_data = {
                    'Model': list(results.keys()),
                    'Accuracy': [results[model]['accuracy'] for model in results.keys()],
                    'Training Time': [results[model]['training_time'] for model in results.keys()]
                }
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    # Accuracy comparison
                    fig_acc = px.bar(
                        x=comparison_data['Model'],
                        y=[acc * 100 for acc in comparison_data['Accuracy']],
                        title="Model Accuracy Comparison (%)",
                        labels={'x': 'Model', 'y': 'Accuracy (%)'},
                        color=comparison_data['Accuracy'],
                        color_continuous_scale='RdYlGn'
                    )
                    fig_acc.update_layout(height=400)
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with comp_col2:
                    # Training time comparison
                    fig_time = px.bar(
                        x=comparison_data['Model'],
                        y=comparison_data['Training Time'],
                        title="Training Time Comparison (seconds)",
                        labels={'x': 'Model', 'y': 'Training Time (s)'},
                        color=comparison_data['Training Time'],
                        color_continuous_scale='Viridis'
                    )
                    fig_time.update_layout(height=400)
                    st.plotly_chart(fig_time, use_container_width=True)
                
                # Performance vs Speed Analysis
                st.subheader("⚡ Performance vs Speed Analysis")
                
                # Create scatter plot
                fig_scatter = px.scatter(
                    x=comparison_data['Training Time'],
                    y=[acc * 100 for acc in comparison_data['Accuracy']],
                    text=comparison_data['Model'],
                    title="Model Performance vs Training Speed",
                    labels={'x': 'Training Time (seconds)', 'y': 'Accuracy (%)'},
                    size=[50] * len(comparison_data['Model']),
                    color=comparison_data['Accuracy'],
                    color_continuous_scale='RdYlGn'
                )
                fig_scatter.update_traces(textposition="top center")
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Model recommendations
                st.subheader("🎯 Model Recommendations")
                
                best_accuracy_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
                fastest_model = min(results.keys(), key=lambda x: results[x]['training_time'])
                
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.info(f"**Best Accuracy**: {best_accuracy_model} ({results[best_accuracy_model]['accuracy']:.1%})")
                    st.write("Recommended for: Production deployment where accuracy is critical")
                
                with rec_col2:
                    st.info(f"**Fastest Training**: {fastest_model} ({results[fastest_model]['training_time']:.2f}s)")
                    st.write("Recommended for: Rapid prototyping and real-time retraining")
                
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")

def show_prediction_interface():
    st.header("🔮 Emotion Prediction Interface")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the Model Training page")
        return
    
    classifier = st.session_state.classifier
    results = st.session_state.training_results
    
    # Model selection
    selected_model = st.selectbox(
        "Select model for prediction:",
        list(results.keys()),
        key="prediction_model"
    )
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Single Text", "Batch Prediction"]
    )
    
    if input_method == "Single Text":
        # Single text prediction
        st.subheader("Single Text Prediction")
        
        user_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type your text here...",
            height=100
        )
        
        if st.button("Predict Emotion") and user_input.strip():
            try:
                with st.spinner("Analyzing emotion..."):
                    prediction, probabilities = classifier.predict_single(user_input, selected_model)
                
                # Display prediction
                st.success(f"Predicted Emotion: **{prediction}**")
                
                # Display probabilities
                st.subheader("Confidence Scores")
                
                prob_df = pd.DataFrame({
                    'Emotion': list(probabilities.keys()),
                    'Probability': list(probabilities.values())
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(
                    prob_df,
                    x='Probability',
                    y='Emotion',
                    orientation='h',
                    title="Emotion Confidence Scores",
                    color='Probability',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    else:
        # Batch prediction
        st.subheader("Batch Prediction")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with 'text' column for batch prediction",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                
                if 'text' not in batch_df.columns:
                    st.error("CSV file must contain a 'text' column")
                else:
                    st.info(f"Loaded {len(batch_df)} texts for prediction")
                    
                    if st.button("Predict Batch"):
                        with st.spinner("Processing batch predictions..."):
                            predictions = []
                            
                            for text in batch_df['text']:
                                if pd.notna(text) and str(text).strip():
                                    pred, _ = classifier.predict_single(str(text), selected_model)
                                    predictions.append(pred)
                                else:
                                    predictions.append('Unknown')
                            
                            batch_df['predicted_emotion'] = predictions
                            
                            st.success("Batch prediction completed!")
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(batch_df)
                            
                            # Download results
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="emotion_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Show prediction distribution
                            pred_counts = pd.Series(predictions).value_counts()
                            fig = px.pie(
                                values=pred_counts.values,
                                names=pred_counts.index,
                                title="Distribution of Predicted Emotions"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
            except Exception as e:
                st.error(f"Error processing batch file: {str(e)}")

if __name__ == "__main__":
    main()
