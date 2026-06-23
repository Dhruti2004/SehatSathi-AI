import os
import shutil
import requests
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, Response, send_file
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferWindowMemory
from deep_translator import GoogleTranslator
import json
import re
from fpdf import FPDF
import datetime

# Load environment variables
load_dotenv()

# ==================== Flask App Setup ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ==================== RAG and LLM Setup ====================
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.environ.get("GROQ_API_KEY"))
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
vectorstore = None
rag_chain = None
user_memories = {}

# ==================== Database Model ====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=True)
    password_hash = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    allergies = db.Column(db.String(300), nullable=True)
    medical_history = db.Column(db.String(500), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

with app.app_context():
    db.create_all()

# ==================== Helper Functions ====================
def create_pdf_report(title, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, 0, 1, "C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    safe_content = content.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, safe_content)
    return Response(
        pdf.output(dest='S').encode('latin-1'),
        mimetype='application/pdf',
        headers={"Content-Disposition": 'attachment;filename=SehatSathiAI_Report.pdf'}
    )

def get_user_memory(user_id):
    if user_id not in user_memories:
        # Keeps last 10 exchanges (20 messages) in memory
        user_memories[user_id] = ConversationBufferWindowMemory(k=10)
    return user_memories[user_id]

def detect_language(text):
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": text[:100]}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        detected = data[2] if len(data) > 2 and data[2] else 'en'
        return detected
    except Exception:
        return 'en'

def translate_to_english(text):
    try:
        user_lang = detect_language(text)
        if user_lang != 'en':
            translated = GoogleTranslator(source=user_lang, target='en').translate(text)
            return translated, user_lang
        return text, 'en'
    except Exception:
        return text, 'en'

def translate_to_user_lang(text, lang):
    try:
        if lang and lang != 'en':
            # Split into smaller chunks to avoid API limit
            chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
            translated_chunks = []
            for chunk in chunks:
                t = GoogleTranslator(source='en', target=lang).translate(chunk)
                if t:
                    translated_chunks.append(t)
                else:
                    translated_chunks.append(chunk)
            return ''.join(translated_chunks)
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def check_drug_interaction(drug1, drug2):
    """
    Check drug interaction via OpenFDA API.
    Returns a dict with interaction_found, reactions, message.
    """
    try:
        url = (
            f"https://api.fda.gov/drug/event.json"
            f"?search=patient.drug.medicinalproduct:{drug1}"
            f"+AND+patient.drug.medicinalproduct:{drug2}&limit=3"
        )
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])
        if results:
            reactions = results[0].get('patient', {}).get('reaction', [])
            reaction_list = [r.get('reactionmeddrapt', '') for r in reactions[:5]]
            return {
                'interaction_found': True,
                'reactions': reaction_list,
                'message': (
                    f"⚠️ Potential interaction found between **{drug1}** and **{drug2}**.\n"
                    f"Reported adverse reactions: {', '.join(reaction_list)}.\n"
                    f"Please consult your doctor before combining these medications."
                )
            }
        return {
            'interaction_found': False,
            'reactions': [],
            'message': (
                f"✅ No known interaction found between **{drug1}** and **{drug2}** "
                f"in the FDA adverse event database. Always verify with your pharmacist."
            )
        }
    except requests.exceptions.Timeout:
        return {'interaction_found': False, 'reactions': [], 'message': "FDA API timed out. Please try again later."}
    except Exception as e:
        return {'interaction_found': False, 'reactions': [], 'message': f"Could not check interaction: {str(e)}"}

def setup_rag_chain_from_file(file_path):
    global vectorstore, rag_chain
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=embedding_model
        )
        rag_prompt_template = """
        SYSTEM PERSONA: You are SehatSathiAI, a prescription analysis expert.
        USER DETAILS: {user_details}
        PRESCRIPTION CONTEXT: {context}
        YOUR TASK: Analyze the provided prescription context based on the user's details. Structure your response with these exact markdown headings:
        **Overall Safety Assessment:**
        **Dosage Check:**
        **Allergy & Interaction Check:**
        **Guidance:**
        """
        rag_prompt = PromptTemplate.from_template(rag_prompt_template)
        rag_chain = (
            {"context": vectorstore.as_retriever(), "user_details": RunnablePassthrough()}
            | rag_prompt | llm | StrOutputParser()
        )
        return True
    except Exception as e:
        print(f"Error setting up RAG chain: {e}")
        return False

# ==================== Routes ====================
@app.route('/')
def root():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'user_id' in session:
        user = db.session.get(User, session['user_id'])
        return render_template('Home.html', user=user)
    return redirect(url_for('login'))

@app.route('/features')
def features():
    if 'user_id' in session:
        user = db.session.get(User, session['user_id'])
        return render_template('Features.html', user=user)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email', '')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return redirect(url_for('register'))
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('Register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home'))
        flash('Invalid username or password.', 'danger')
    return render_template('Login.html')

@app.route('/logout')
def logout():
    user_id = session.get('user_id')
    if user_id and user_id in user_memories:
        del user_memories[user_id]
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/assistant')
def assistant():
    if 'user_id' in session:
        user = db.session.get(User, session['user_id'])
        session['chat_stage'] = 'GENERAL'
        history = ChatHistory.query.filter_by(user_id=session['user_id']).order_by(ChatHistory.timestamp).all()
        return render_template('Index.html', user=user, history=history)
    return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.json
    user_input = data.get('message')
    is_prediction_start = data.get('is_prediction_start', False)
    user = db.session.get(User, session['user_id'])
    user_profile = f"Age: {user.age or 'Not provided'}, Allergies: {user.allergies or 'Not provided'}, Medical History: {user.medical_history or 'Not provided'}"
    chat_stage = session.get('chat_stage', 'GENERAL')

    english_input, user_lang = translate_to_english(user_input)
    session['user_lang'] = user_lang

    memory = get_user_memory(session['user_id'])

    if is_prediction_start:
        chat_stage = 'PREDICTION_STARTED'
        session['chat_stage'] = chat_stage
        session['symptom_data'] = {'initial': user_input, 'answers': []}

    try:
        ai_response = ""
        if chat_stage == 'PREDICTION_STARTED':
            prompt = f"""You are a medical data gathering AI. A user has reported these symptoms: "{session['symptom_data']['initial']}". 
                        Generate exactly 5 follow-up questions as a JSON array.
                        Return ONLY the JSON array, no extra text, no markdown, no explanation.
                        Example format: ["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]"""
            response_text = llm.invoke(prompt).content.strip()
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            try:
                questions = json.loads(response_text)
            except:
                questions = [
                    "How long have you been experiencing these symptoms?",
                    "Have you taken any medication for these symptoms?",
                    "Are there any activities that make the symptoms better or worse?",
                    "Do you have any other health conditions?",
                    "When did you first notice these symptoms?"
                ]
            session['follow_up_questions'] = questions
            session['chat_stage'] = 'AWAITING_ANSWER_1'
            ai_response = questions[0]

        elif chat_stage == 'AWAITING_ANSWER_1':
            session['symptom_data']['answers'].append(user_input)
            session['chat_stage'] = 'AWAITING_ANSWER_2'
            ai_response = session['follow_up_questions'][1]

        elif chat_stage == 'AWAITING_ANSWER_2':
            session['symptom_data']['answers'].append(user_input)
            session['chat_stage'] = 'AWAITING_ANSWER_3'
            ai_response = session['follow_up_questions'][2]

        elif chat_stage == 'AWAITING_ANSWER_3':
            session['symptom_data']['answers'].append(user_input)
            s_data = session['symptom_data']
            full_context = f"Initial Symptoms: {s_data['initial']}. Answers: 1. {s_data['answers'][0]}, 2. {s_data['answers'][1]}, 3. {s_data['answers'][2]}"
            prompt = f"""
            SYSTEM PERSONA: You are SehatSathiAI, a virtual first doctor.
            IMPORTANT: Always respond in clear, simple English only. Do NOT mix any other language words. Use only pure English words throughout your response.
            USER DATA: Profile: {user_profile}. Full Symptom Report: {full_context}.
            YOUR TASK: Based on all user data, generate a structured medical report with these exact markdown headings:
            **Disclaimer:** (Warn that you are an AI and not a substitute for a real doctor.)
            **Possible Illness(es):** (List 1-2 likely conditions.)
            **Recommended Generic Medicines:** (Suggest over-the-counter medicines like Paracetamol with dosage.)
            **Lifestyle and Home Care:** (Provide a bulleted list of advice.)
            **When to See A Doctor:** (List critical symptoms that require immediate medical attention.)
            """
            ai_response = llm.invoke(prompt).content
            session['last_prediction_result'] = ai_response
            session.pop('last_analysis_result', None)
            session['chat_stage'] = 'GENERAL'
            session.pop('symptom_data', None)
            session.pop('follow_up_questions', None)

        else:
            # General conversation with memory
            history = memory.load_memory_variables({}).get('history', '')
            prompt = f"""You are SehatSathiAI, a helpful medical assistant.
            IMPORTANT: Always respond in clear, simple English only. Do NOT mix any other language words into your English response. Use only pure English words.
            User profile: {user_profile}
            Conversation history: {history}
            User question: '{english_input}'
            Answer helpfully and informatively using only pure English words."""
            ai_response = llm.invoke(prompt).content
            memory.save_context(
                {"input": english_input},
                {"output": ai_response}
            )

        ai_response = translate_to_user_lang(ai_response, user_lang)
        entry = ChatHistory(
            user_id=session['user_id'],
            question=user_input,
            answer=ai_response
        )
        db.session.add(entry)
        db.session.commit()

        return jsonify({'response': ai_response})

    except Exception as e:
        import traceback
        traceback.print_exc()
        session['chat_stage'] = 'GENERAL'
        return jsonify({'error': str(e)}), 500

# ── NEW: Drug Interaction Checker endpoint ──
@app.route('/check-drug-interaction', methods=['POST'])
def drug_interaction():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json()
    drug1 = data.get('drug1', '').strip()
    drug2 = data.get('drug2', '').strip()
    if not drug1 or not drug2:
        return jsonify({'error': 'Please provide both drug names'}), 400
    result = check_drug_interaction(drug1, drug2)
    # Save to chat history so user can download it later
    entry = ChatHistory(
        user_id=session['user_id'],
        question=f"Drug interaction check: {drug1} + {drug2}",
        answer=result['message']
    )
    db.session.add(entry)
    db.session.commit()
    return jsonify(result)

@app.route('/get_user_info', methods=['GET'])
def get_user_info():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user = db.session.get(User, session['user_id'])
    if user:
        return jsonify({
            'username': user.username,
            'email': user.email or '',
            'age': user.age,
            'allergies': user.allergies or '',
            'medical_history': user.medical_history or ''
        })
    return jsonify({'error': 'User not found'}), 404

@app.route('/update_user_info', methods=['POST'])
def update_user_info():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json()
    user = db.session.get(User, session['user_id'])
    if user:
        user.age = data.get('age', user.age)
        user.allergies = data.get('allergies', user.allergies)
        user.medical_history = data.get('medical_history', user.medical_history)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    return jsonify({'error': 'User not found'}), 404

@app.route('/analyze-prescription', methods=['POST'])
def api_analyze_prescription():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files allowed'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    setup_rag_chain_from_file(filepath)
    user = db.session.get(User, session['user_id'])
    user_details = f"""
    Age: {user.age}
    Allergies: {user.allergies}
    Medical History: {user.medical_history}
    """
    try:
        result = rag_chain.invoke(user_details)
        session['last_analysis_result'] = result
        entry = ChatHistory(
            user_id=session['user_id'],
            question=f"Prescription analysis: {file.filename}",
            answer=result
        )
        db.session.add(entry)
        db.session.commit()
        return jsonify({'answer': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-pdf', methods=['GET'])
def download_pdf():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    prediction = session.get(
        'last_prediction_result',
        session.get('last_analysis_result', 'No report available.')
    )
    return create_pdf_report("SehatSathiAI Medical Report", prediction)

@app.route('/clear-chat')
def clear_chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    ChatHistory.query.filter_by(user_id=session['user_id']).delete()
    db.session.commit()
    user_id = session.get('user_id')
    if user_id and user_id in user_memories:
        del user_memories[user_id]
    return redirect(url_for('assistant'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)