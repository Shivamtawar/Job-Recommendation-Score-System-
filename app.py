from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import re
import os
from pdfminer.high_level import extract_text
from docx import Document
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (OSError, ImportError):
    print("Spacy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_file(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith('.pdf'):
        return extract_text(file_path)
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")

def extract_name(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for i, line in enumerate(lines[:10]):
        if len(line) < 3:
            continue
            
        skip_patterns = [
            r'^(resume|cv|curriculum|vitae|profile|summary|objective|contact)',
            r'^(experience|education|skills|projects|achievements)',
            r'^(email|phone|address|linkedin|github)',
            r'@.*\.(com|org|in|net)',
            r'^\+?\d[\d\s\-\(\)]+$',
            r'^https?://',
            r'^\d+\s',
        ]
        
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
            continue
        
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$',
            r'^([A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+)$',
            r'^([A-Z][A-Z\s]{2,30})$',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, line.strip())
            if match:
                name_candidate = match.group(1).strip()
                
                words = name_candidate.split()
                if (len(words) >= 2 and len(words) <= 4 and 
                    all(len(word) > 1 for word in words) and
                    not any(word.lower() in ['engineer', 'developer', 'manager', 'lead', 'senior'] for word in words)):
                    return name_candidate
    
    if nlp:
        doc = nlp(text[:1000])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name_parts = ent.text.split()
                if len(name_parts) >= 2 and len(name_parts) <= 4:
                    if all(len(part) > 1 and part[0].isupper() for part in name_parts):
                        return ent.text
    
    return "Unknown Candidate"

def extract_experience(text):
    experience_data = {
        'experience_list': [],
        'experience_years': 0
    }
    
    experience_patterns = [
        r'with\s+over\s+(\d+(?:\.\d+)?)\s*years?\s*of\s*(?:ex|experience)',
        r'(?:over\s+)?(\d+(?:\.\d+)?)\s*years?\s*of\s*(?:professional\s*)?experience',
        r'(\d+(?:\.\d+)?)\s*years?\s*(?:of\s*)?(?:professional\s*)?experience',
        r'I\s*(?:am|have).*?(?:with\s*)?(\d+(?:\.\d+)?)\s*years?\s*(?:of\s*)?experience',
        r'Lead.*?with.*?(\d+(?:\.\d+)?)\s*years?\s*of\s*(?:ex|experience)',
        r'Engineer.*?with.*?(\d+(?:\.\d+)?)\s*years?\s*of\s*(?:ex|experience)',
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        for match in matches:
            try:
                years = float(match)
                if 0 < years <= 50:
                    experience_data['experience_years'] = max(experience_data['experience_years'], years)
            except ValueError:
                continue
    
    if experience_data['experience_years'] == 0:
        date_patterns = [
            r'((?:Present|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4})\s*↑\s*((?:Present|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4})\s*[-–—↑to]\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|Present|Current)',
            r'(\d{4})\s*[-–—↑to]\s*(\d{4}|Present|Current)',
            r'(\d{1,2}/\d{4})\s*[-–—↑to]\s*(\d{1,2}/\d{4}|Present|Current)',
        ]
        
        total_experience = 0
        current_year = 2024
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for start_date, end_date in matches:
                try:
                    start_year_match = re.search(r'(\d{4})', start_date)
                    if start_year_match:
                        start_year = int(start_year_match.group(1))
                        
                        if end_date.lower() in ['present', 'current'] or 'Present' in end_date:
                            end_year = current_year
                        else:
                            end_year_match = re.search(r'(\d{4})', end_date)
                            if end_year_match:
                                end_year = int(end_year_match.group(1))
                            else:
                                continue
                        
                        if start_year <= end_year <= current_year and start_year >= 2000:
                            years_worked = end_year - start_year
                            if years_worked > 0:
                                total_experience += years_worked
                except Exception:
                    continue
        
        if total_experience > 0:
            experience_data['experience_years'] = total_experience
    
    return experience_data

def extract_skills(text):
    skills = set()
    
    tech_patterns = [
        r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin|Scala|R)\b',
        r'\b(TensorFlow|PyTorch|Keras|Scikit-learn|sklearn|Pandas|NumPy|Matplotlib|Seaborn)\b',
        r'\b(Hugging\s*Face|Transformers|BERT|RoBERTa|GPT|LLM|LangChain|LangGraph)\b',
        r'\b(OpenAI|Anthropic|Llama|Mistral|Claude|CrewAI|Autogen)\b',
        r'\b(Machine\s*Learning|Deep\s*Learning|Neural\s*Networks|NLP|Computer\s*Vision)\b',
        r'\b(Classification|Regression|Clustering|Reinforcement\s*Learning|RLHF|DPO|QLORA)\b',
        r'\b(Recommender\s*Systems|RAG|Retrieval\s*Augmented\s*Generation)\b',
        r'\b(AWS|Azure|GCP|Google\s*Cloud|Amazon\s*Web\s*Services)\b',
        r'\b(SageMaker|Sagemakers|Azure\s*ML|Vertex\s*AI|Databricks|Runpod)\b',
        r'\b(AWS\s*Glue|Glue)\b',
        r'\b(MySQL|PostgreSQL|Postgres|MongoDB|Redis|Elasticsearch|Neo4j|Cassandra)\b',
        r'\b(Vector\s*Database|FAISS|Pinecone|Weaviate)\b',
        r'\b(Flask|FastAPI|Django|Express|React|Angular|Vue|Node\.js)\b',
        r'\b(Docker|Kubernetes|Jenkins|Git|GitHub|GitLab|CI/CD)\b',
        r'\b(Spark|Kafka|Airflow|ETL|Data\s*Pipeline)\b',
        r'\b(Tableau|Power\s*BI|D3\.js|Plotly|Streamlit)\b',
        r'\b(OpenCV|YOLO|SAM2|opencv)\b',
        r'\b(llamaindex|Dspy|GraphRAG|Phi\s*data)\b',
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, str) and len(match.strip()) > 1:
                cleaned_skill = match.strip()
                if not any(word in cleaned_skill.lower() for word in ['fitness', 'gaming', 'english', 'hindi', 'fluent']):
                    skills.add(cleaned_skill)
    
    skills_section_patterns = [
        r'languages\s*[:\n](.*?)(?=Libraries|$)',
        r'Libraries\s*[:\n](.*?)(?=Cloud|$)',  
        r'Cloud\s*[:\n](.*?)(?=Databases|$)',
        r'Databases\s*[:\n](.*?)(?=Certifications|$)',
        r'(?:Technical\s+)?Skills?\s*[:\n]\s*(.*?)(?=\n\s*(?:Experience|Education|Projects|Achievements|Contact)\s*[:|\n]|$)',
        r'Technologies?\s*[:\n]\s*(.*?)(?=\n\s*(?:Experience|Education|Projects|Achievements|Contact)\s*[:|\n]|$)',
    ]
    
    for pattern in skills_section_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            skills_section = match.group(1)
            
            bullet_skills = re.findall(r'○\s*([^○\n]+)', skills_section)
            for skill_line in bullet_skills:
                if any(word in skill_line.lower() for word in ['fitness', 'gaming', 'football', 'bike', 'gym', 'communication', 'teamwork', 'english', 'hindi', 'fluent']):
                    continue
                
                individual_skills = [s.strip() for s in skill_line.split(',') if s.strip()]
                for skill in individual_skills:
                    if len(skill) > 1 and len(skill) < 40:
                        skills.add(skill)
    
    work_skills = re.findall(r'\b(LangGraph|langchain|RAG|LLM|QLORA|DPO|Autogen|CrewAI|OpenAI|Hugging\s*Face|AWS\s*Glue|Sagemakers|OCR|BERT|RoBERTa|YOLO|SAM2)\b', text, re.IGNORECASE)
    for skill in work_skills:
        if skill.strip():
            skills.add(skill.strip())
    
    cleaned_skills = []
    for skill in skills:
        skill = skill.strip()
        if (skill and len(skill) > 1 and len(skill) < 40 and
            not any(word in skill.lower() for word in ['fitness', 'gaming', 'football', 'bike', 'gym', 
                                                       'communication', 'teamwork', 'management', 'solving',
                                                       'english', 'hindi', 'fluent', 'riding'])):
            cleaned_skills.append(skill)
    
    return list(set(cleaned_skills))[:25]

def extract_education(text):
    education_patterns = [
        r'Education\s*[:\n]\s*(.*?)(?=\n\s*(?:Experience|Skills|Projects|Achievements|Contact)\s*[:|\n]|$)',
        r'Academic\s+Background\s*[:\n]\s*(.*?)(?=\n\s*(?:Experience|Skills|Projects|Achievements|Contact)\s*[:|\n]|$)',
        r'Qualifications?\s*[:\n]\s*(.*?)(?=\n\s*(?:Experience|Skills|Projects|Achievements|Contact)\s*[:|\n]|$)',
        r'\b(Bachelor(?:\s+of\s+Technology)?|B\.?Tech|Master(?:\s+of\s+Technology)?|M\.?Tech|PhD|Doctorate)\b[^\n]{0,100}',
        r'\b(B\.?E\.?|M\.?E\.?|B\.?S\.?c|M\.?S\.?c|BCA|MCA|MBA)\b[^\n]{0,100}',
    ]
    
    education_info = ""
    for pattern in education_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            education_info += " " + match.group(1 if 'Education' in pattern else 0).strip()
    
    return education_info.strip()

def parse_resume(file_path):
    text = extract_text_from_file(file_path)
    
    name = extract_name(text)
    experience_data = extract_experience(text)
    skills = extract_skills(text)
    education = extract_education(text)
    
    return {
        'name': name,
        'experience': experience_data['experience_list'],
        'experience_years': experience_data['experience_years'],
        'skills': skills,
        'education': education
    }

def parse_jd(jd_text):    
    role_patterns = [
        r'Job\s+Title\s*[:\-]\s*(.+?)(?:\n|$)',
        r'Role\s*[:\-]\s*(.+?)(?:\n|$)', 
        r'Position\s*[:\-]\s*(.+?)(?:\n|$)',
        r'^(.+?(?:Engineer|Developer|Manager|Analyst|Designer|Scientist|Specialist|Lead))(?:\n|$)',
    ]
    
    role = "Unknown Position"
    for pattern in role_patterns:
        role_match = re.search(pattern, jd_text, re.IGNORECASE | re.MULTILINE)
        if role_match:
            role = role_match.group(1).strip()
            break
    
    exp_patterns = [
        r'(\d+)(?:\s*[-–—+to]\s*\d+)?\+?\s*years?\s*of\s*experience',
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'minimum\s*(?:of\s*)?(\d+)\s*years?',
        r'at\s*least\s*(\d+)\s*years?',
        r'(\d+)(?:\s*[-–—]\s*\d+)?\s*(?:\+)?\s*years?\s*(?:in|with)',
    ]
    
    required_experience_years = 0
    for pattern in exp_patterns:
        exp_match = re.search(pattern, jd_text, re.IGNORECASE)
        if exp_match:
            try:
                required_experience_years = int(exp_match.group(1))
                break
            except ValueError:
                continue
    
    required_skills = set()
    
    comprehensive_tech_patterns = [
        r'\b(scikit-learn|TensorFlow|PyTorch|Transformers|Hugging\s*Face)\b',
        r'\b(LangChain|LangGraph|FastAPI|Flask|Django|React|Angular|Vue\.js|Node\.js)\b',
        r'\b(Pandas|NumPy|Matplotlib|Seaborn|OpenCV|Pillow)\b',
        r'\b(Amazon\s*Web\s*Services|AWS|Microsoft\s*Azure|Azure|Google\s*Cloud|GCP)\b',
        r'\b(Docker|Kubernetes|Jenkins|Git|GitHub|GitLab)\b',
        r'\b(MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Neo4j)\b',
        r'\b(FAISS|Pinecone|Weaviate|Vector\s*Database|vector\s*databases)\b',
        r'\b(Machine\s*Learning|Deep\s*Learning|Natural\s*Language\s*Processing|NLP)\b',
        r'\b(Computer\s*Vision|Recommender\s*Systems|Classification\s*Models)\b',
        r'\b(BERT|SBERT|GPT|LLM|Large\s*Language\s*Models)\b',
        r'\b(Retrieval\s*Augmented\s*Generation|RAG)\b',
        r'\b(Python|JavaScript|Java|C\+\+|SQL|R|Scala)\b',
        r'\b(Jupyter|Git|Streamlit|Tableau|Power\s*BI)\b',
        r'\b(embeddings|text\s*similarity|resume\s*parsing)\b',
    ]
    
    for pattern in comprehensive_tech_patterns:
        matches = re.findall(pattern, jd_text, re.IGNORECASE)
        for match in matches:
            if match.strip():
                required_skills.add(match.strip())
    
    tech_stack_patterns = [
        r'Tech\s+Stack\s*[:\n](.*?)(?=Requirements|What\s+You|$)',
        r'Technologies?\s*[:\n](.*?)(?=Requirements|What\s+You|$)',
        r'Technical\s+Requirements?\s*[:\n](.*?)(?=Requirements|What\s+You|$)',
    ]
    
    for pattern in tech_stack_patterns:
        tech_match = re.search(pattern, jd_text, re.DOTALL | re.IGNORECASE)
        if tech_match:
            tech_section = tech_match.group(1)
            
            lines = tech_section.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('•'):
                    continue
                
                category_match = re.search(r'^[A-Za-z/\s]+:\s*(.+)', line)
                if category_match:
                    items_text = category_match.group(1)
                    items = [item.strip() for item in re.split(r',(?![^()]*\))', items_text)]
                    for item in items:
                        if item and len(item) > 1 and len(item) < 50:
                            cleaned_item = re.sub(r'\s*\([^)]*\)', '', item).strip()
                            if cleaned_item:
                                required_skills.add(cleaned_item)
    
    req_patterns = [
        r'Requirements?\s*[:\n](.*?)(?=\n[A-Z][a-z]+\s*[:|\n]|$)',
        r'(?:Must\s+Have|Required)\s*[:\n](.*?)(?=\n[A-Z][a-z]+\s*[:|\n]|$)',
    ]
    
    for pattern in req_patterns:
        req_match = re.search(pattern, jd_text, re.DOTALL | re.IGNORECASE)
        if req_match:
            req_section = req_match.group(1)
            
            for tech_pattern in comprehensive_tech_patterns:
                matches = re.findall(tech_pattern, req_section, re.IGNORECASE)
                for match in matches:
                    if match.strip():
                        required_skills.add(match.strip())
    
    cleaned_skills = []
    for skill in required_skills:
        skill = skill.strip()
        if (skill and len(skill) > 2 and len(skill) < 50 and 
            not skill.lower() in ['strong', 'experience', 'familiarity', 'ability', 'prior', 'bonus', 'jds', 'mvp'] and
            not skill.endswith(')')):
            cleaned_skills.append(skill)
    
    education_patterns = [
        r"(?:Bachelor'?s?|Master'?s?|PhD|Doctorate).*?(?:degree|in|of)\s*([^.\n,]+)",
        r"\b(?:Bachelor|Master|PhD|Doctorate)\b.*?(?:degree|in|of)?\s*([^.\n,]+)"
    ]
    
    required_education = ""
    for pattern in education_patterns:
        edu_match = re.search(pattern, jd_text, re.IGNORECASE)
        if edu_match:
            required_education = edu_match.group(0).strip()
            break
    
    return {
        'title': role,
        'required_experience_years': required_experience_years,
        'required_skills': cleaned_skills[:20],
        'required_education': required_education
    }

def calculate_skill_similarity(candidate_skill, required_skill):    
    candidate_lower = candidate_skill.lower().strip()
    required_lower = required_skill.lower().strip()
    
    if candidate_lower == required_lower:
        return 95
    
    if candidate_lower in required_lower or required_lower in candidate_lower:
        return 85
    
    try:
        emb1 = model.encode([candidate_skill])
        emb2 = model.encode([required_skill])
        semantic_similarity = util.cos_sim(emb1, emb2).item()
        
        if semantic_similarity > 0.9:
            return 90
        elif semantic_similarity > 0.8:
            return 80
        elif semantic_similarity > 0.7:
            return 70
        elif semantic_similarity > 0.6:
            return 60
        elif semantic_similarity > 0.5:
            return 50
        else:
            return semantic_similarity * 60
            
    except Exception:
        candidate_words = set(candidate_lower.split())
        required_words = set(required_lower.split())
        
        if not required_words:
            return 0
            
        intersection = len(candidate_words.intersection(required_words))
        union = len(candidate_words.union(required_words))
        
        if union == 0:
            return 0
            
        jaccard_similarity = intersection / union
        return min(85, jaccard_similarity * 100)

def calculate_scores(resume_data, jd_data):    
    candidate_years = resume_data['experience_years']
    required_years = jd_data['required_experience_years']
    
    if required_years > 0:
        if candidate_years >= required_years:
            ratio = candidate_years / required_years
            if ratio >= 2.0:
                experience_match = 95
            elif ratio >= 1.5:
                experience_match = 90
            else:
                experience_match = 85
        else:
            ratio = candidate_years / required_years
            if ratio >= 0.8:
                experience_match = 75
            elif ratio >= 0.6:
                experience_match = 60
            else:
                experience_match = max(20, ratio * 70)
    else:
        experience_match = 80 if candidate_years > 0 else 30
    
    if not jd_data['required_skills'] or not resume_data['skills']:
        skills_match = 0
    else:
        skill_scores = []
        
        for req_skill in jd_data['required_skills']:
            best_score = 0
            for candidate_skill in resume_data['skills']:
                score = calculate_skill_similarity(candidate_skill, req_skill)
                best_score = max(best_score, score)
            skill_scores.append(best_score)
        
        if skill_scores:
            skills_match = np.mean(skill_scores)
        else:
            skills_match = 0
    
    if not resume_data['education'] or not jd_data['required_education']:
        education_match = 70 if resume_data['education'] else 50
    else:
        candidate_edu = resume_data['education'].lower()
        required_edu = jd_data['required_education'].lower()
        
        has_bachelor = any(term in candidate_edu for term in ['bachelor', 'b.tech', 'b.sc', 'be'])
        has_master = any(term in candidate_edu for term in ['master', 'm.tech', 'm.sc', 'me'])
        has_phd = any(term in candidate_edu for term in ['phd', 'doctorate'])
        
        requires_bachelor = 'bachelor' in required_edu
        requires_master = 'master' in required_edu
        requires_phd = 'phd' in required_edu or 'doctorate' in required_edu
        
        if requires_phd and has_phd:
            education_match = 95
        elif requires_master and (has_master or has_phd):
            education_match = 92
        elif requires_bachelor and (has_bachelor or has_master or has_phd):
            education_match = 88
        elif has_bachelor or has_master or has_phd:
            education_match = 75
        else:
            education_match = 40
    
    overall_score = (skills_match * 0.5) + (experience_match * 0.3) + (education_match * 0.2)
    
    return {
        'experience_match': round(experience_match),
        'skills_match': round(skills_match),
        'education_match': round(education_match),
        'overall_score': round(overall_score)
    }

@app.route('/match', methods=['POST'])
def match():
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file'}), 400
    resume_file = request.files['resume']
    jd_text = request.form.get('jd', '')

    if not jd_text:
        return jsonify({'error': 'No job description provided'}), 400

    resume_path = 'temp_resume.' + resume_file.filename.split('.')[-1]
    resume_file.save(resume_path)

    try:
        resume_data = parse_resume(resume_path)
        jd_data = parse_jd(jd_text)
        scores = calculate_scores(resume_data, jd_data)

        response = {
            'candidate_name': resume_data['name'],
            'job_title': jd_data['title'],
            'match_scores': {
                'experience_match': scores['experience_match'],
                'skills_match': scores['skills_match'], 
                'education_match': scores['education_match'],
                'overall_score': scores['overall_score']
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(resume_path):
            os.remove(resume_path)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if 'resume' not in request.files:
        return "No resume file", 400
    resume_file = request.files['resume']
    jd_text = request.form.get('jd', '')

    resume_path = 'temp_resume.' + resume_file.filename.split('.')[-1]
    resume_file.save(resume_path)

    try:
        resume_data = parse_resume(resume_path)
        jd_data = parse_jd(jd_text)
        scores = calculate_scores(resume_data, jd_data)
        return render_template('result.html', response={
            'candidate_name': resume_data['name'],
            'job_title': jd_data['title'],
            'match_scores': scores,
            'debug_info': {
                'extracted_skills': resume_data['skills'][:15],
                'extracted_years': resume_data['experience_years'],
                'required_skills': jd_data['required_skills'][:10],
                'required_years': jd_data['required_experience_years']
            }
        })
    except Exception as e:
        return f"Error processing resume: {str(e)}", 500
    finally:
        if os.path.exists(resume_path):
            os.remove(resume_path)

if __name__ == '__main__':
    app.run(debug=True)