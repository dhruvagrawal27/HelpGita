import streamlit as st
import time

# Page setup
st.set_page_config(
    page_title="KrishnaGuides - Spiritual Wisdom",
    page_icon="🕉️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Serif';
        text-align: center;
        color: #5E35B1;
    }
    .subheader {
        text-align: center;
        color: #7E57C2;
        margin-bottom: 25px;
    }
    .price-card {
        background-color: #F3F0FF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 30px;
    }
    .highlight {
        color: #5E35B1;
        font-weight: bold;
    }
    .feature-item {
        margin: 10px 0;
    }
    .plan-name {
        font-size: 24px;
        font-weight: bold;
        color: #5E35B1;
    }
    .price-text {
        font-size: 36px;
        font-weight: bold;
        margin: 15px 0;
        color: #4527A0;
    }
    .cta-button {
        background-color: #5E35B1;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        border: none;
        cursor: pointer;
        width: 100%;
        font-size: 18px;
    }
    .testimonial {
        font-style: italic;
        background-color: #F0F0F0;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .faq-question {
        font-weight: bold;
        color: #5E35B1;
        cursor: pointer;
        margin-top: 15px;
    }
    .lotus-divider {
        text-align: center;
        font-size: 24px;
        margin: 20px 0;
        color: #7E57C2;
    }
    .banner {
        padding: 12px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 20px;
        background-color: #EDE7F6;
        border-left: 5px solid #7E57C2;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">KrishnaGuides</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subheader">Ancient Wisdom for Modern Questions</h3>', unsafe_allow_html=True)

# Hero Image
st.image("image.jpeg", use_container_width=True)

# Toggle for monthly/yearly
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    billing_option = st.toggle("Monthly / Annual (Save 20%)", value=False)
    price = "$49/month" if not billing_option else "$469/year"
    billed_text = "Billed monthly" if not billing_option else "Billed annually (save $119)"

# Main pricing card
st.markdown('<div class="price-card">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('<div class="plan-name">Spiritual Seeker</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="price-text">{price}</div>', unsafe_allow_html=True)
    st.markdown(f'<div>{billed_text}</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="highlight">What You Get:</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-item">✅ Unlimited spiritual questions</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-item">✅ Bhagavad Gita-based guidance</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-item">✅ 24/7 wisdom access</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-item">✅ Cancel anytime</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sample Questions
st.markdown("## Ask Your Spiritual Question")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Responses dictionary
responses = {
    "How can I overcome fear based on Gita?": "According to the Bhagavad Gita, fear arises from attachment and ignorance of our true nature. Krishna teaches Arjuna that the soul is eternal and can neither be born nor die. In Chapter 2, Verse 14, Krishna says: 'The temporary appearance of happiness and distress, and their disappearance in due course, are like the appearance and disappearance of winter and summer seasons. They arise from sense perception, and one must learn to tolerate them without being disturbed.' By understanding our true spiritual identity and practicing equanimity, we can transcend fear.",
    
    "What does Krishna say about purpose?": "Krishna explains in the Gita that each person has a unique purpose (dharma) aligned with their natural qualities. In Chapter 3, Verse 35, he states: 'It is better to perform one's own duties imperfectly than to master the duties of another. By fulfilling the obligations born of one's nature, one never incurs sin.' Your purpose involves both your natural talents and performing your actions as service without attachment to rewards, as taught in Chapter 2, Verse 47: 'You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions.'",
    
    "How to find inner peace?": "The Bhagavad Gita offers several paths to inner peace. Krishna emphasizes the importance of steady mind through meditation (dhyana yoga), selfless action without attachment to results (karma yoga), and devotion (bhakti yoga). In Chapter 6, Verse 7, Krishna explains: 'For one who has conquered the mind, the mind is the best of friends; but for one who has failed to do so, the mind will remain the greatest enemy.' Regular meditation, performing actions without attachment to results, and maintaining equanimity in success and failure are key Gita practices for finding lasting inner peace.",
    
    "Dealing with stress through spirituality": "The Bhagavad Gita offers profound wisdom for stress management. Krishna advises Arjuna to practice detachment from outcomes while remaining fully engaged in action. In Chapter 2, Verse 48, he says: 'Perform your duty equipoised, abandoning all attachment to success or failure.' The Gita also recommends meditation to calm the mind, as stated in Chapter 6, Verse 27: 'The yogi whose mind is fixed on Me verily attains the highest tranquility.' By focusing on the present moment, offering your actions as service, and maintaining spiritual awareness, you can reduce stress and find greater equilibrium."
}

default_response = "The Bhagavad Gita teaches that spiritual knowledge is the path to inner peace and fulfillment. Krishna says in Chapter 4, Verse 39: 'One who has faith, who is dedicated and who has mastered his senses, attains this knowledge. Having attained this knowledge, one quickly attains supreme peace.' I'd be happy to explore your question in more depth - please start your subscription to continue our conversation."

# Function to handle sample question clicks
def handle_sample_question(question):
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Check if trial has started, if not set default message
    if not st.session_state.get("trial_started", False):
        answer = "To receive guidance based on the Bhagavad Gita's wisdom, please start your subscription."
    else:
        answer = responses.get(question, default_response)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# Sample questions user can click
st.markdown("### Need inspiration? Try one of these questions:")
col1, col2 = st.columns(2)
with col1:
    if st.button("How can I overcome fear based on Gita?"):
        handle_sample_question("How can I overcome fear based on Gita?")
        
    if st.button("What does Krishna say about purpose?"):
        handle_sample_question("What does Krishna say about purpose?")

with col2:
    if st.button("How to find inner peace?"):
        handle_sample_question("How to find inner peace?")
        
    if st.button("Dealing with stress through spirituality"):
        handle_sample_question("Dealing with stress through spirituality")

# Testimonials
st.markdown('<div class="lotus-divider"></div>', unsafe_allow_html=True)
st.markdown("## What Our Members Say")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="testimonial">"KrishnaGuides has transformed my approach to daily challenges. The spiritual wisdom from the Gita has given me a new perspective on lifes difficulties." <br>- Priya M.</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="testimonial">"I turn to this service whenever I feel overwhelmed. The timeless wisdom helps me center myself and find clarity." <br>- Michael K.</div>', unsafe_allow_html=True)

# FAQ Section
st.markdown('<div class="lotus-divider"></div>', unsafe_allow_html=True)
st.markdown("## Frequently Asked Questions")

faq_items = {
    "What is KrishnaGuides?": "KrishnaGuides is a spiritual Q&A service providing wisdom and guidance based on the teachings of the Bhagavad Gita. It helps you navigate modern challenges using ancient spiritual principles.",
    
    "Do I need prior knowledge of the Bhagavad Gita?": "Not at all! KrishnaGuides is designed for everyone, from beginners to those well-versed in spirituality. Our answers explain concepts clearly and relate them to everyday situations.",
   
    "Is my personal information stored?": "We prioritize your privacy. We don't store your question history. Each session is private and separate.",
    
    "Can I ask about personal problems?": "Yes! While our guidance is based on spiritual principles from the Bhagavad Gita rather than personal advice, many find these teachings helpful when applied to specific situations."
}

for question, answer in faq_items.items():
    with st.expander(question):
        st.write(answer)