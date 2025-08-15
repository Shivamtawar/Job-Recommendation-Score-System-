from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import re
import os
from pdfminer.high_level import extract_text
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load pre-trained model (downloads automatically on first run, runs locally)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to parse resume (TXT, PDF, DOCX)
def parse_resume(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif file_path.endswith('.pdf'):
        text = extract_text(file_path)
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")

    # Improved parsing with better regex patterns
    # Extract name
    name_match = re.search(r'Name:\s*(.+)', text, re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else "Unknown"
    
    # Extract experience section - look for bullet points and years
    experience_section = re.search(r'Experience:\s*(.*?)(?=Skills:|Education:|$)', text, re.DOTALL | re.IGNORECASE)
    experience = []
    experience_years = 0
    
    if experience_section:
        exp_text = experience_section.group(1).strip()
        # Split by bullet points or line breaks
        exp_lines = [line.strip() for line in re.split(r'[-•]\s*', exp_text) if line.strip()]
        experience = exp_lines
        
        # Extract years of experience more accurately
        year_matches = re.findall(r'(\d+)\s*years?', exp_text, re.IGNORECASE)
        if year_matches:
            experience_years = max([int(year) for year in year_matches])
    
    # Extract skills - handle comma-separated and bullet-point formats
    skills_section = re.search(r'Skills:\s*(.*?)(?=Education:|$)', text, re.DOTALL | re.IGNORECASE)
    skills = []
    
    if skills_section:
        skills_text = skills_section.group(1).strip()
        # Handle both comma-separated and bullet-point formats
        if ',' in skills_text:
            skills = [skill.strip() for skill in re.split(r'[,\n]', skills_text) if skill.strip()]
        else:
            # Handle bullet points
            skills = [skill.strip() for skill in re.split(r'[-•]\s*', skills_text) if skill.strip()]
        
        # Clean up skills (remove dashes, extra whitespace and empty entries)
        cleaned_skills = []
        for skill in skills:
            if skill:
                # Remove leading dashes and clean up
                cleaned_skill = re.sub(r'^[-•]\s*', '', skill).strip()
                if cleaned_skill and len(cleaned_skill) > 1:
                    cleaned_skills.append(cleaned_skill)
        skills = cleaned_skills
    
    # Extract education
    education_section = re.search(r'Education:\s*(.*?)$', text, re.DOTALL | re.IGNORECASE)
    education = ""
    if education_section:
        education = education_section.group(1).strip()
        # Clean up education text
        education = re.sub(r'[-•]\s*', '', education).strip()

    return {
        'name': name,
        'experience': experience,
        'experience_years': experience_years,
        'skills': skills,
        'education': education
    }

# Helper function to parse JD with improved regex
def parse_jd(jd_text):
    # Extract role/title
    role_match = re.search(r'Role:\s*(.+)', jd_text, re.IGNORECASE)
    role = role_match.group(1).strip() if role_match else "Unknown"
    
    # Extract required experience years
    exp_match = re.search(r'(\d+)\+?\s*years?\s*experience', jd_text, re.IGNORECASE)
    required_experience_years = int(exp_match.group(1)) if exp_match else 0
    
    # Extract required skills - look for skills after "Required:" section
    required_section = re.search(r'Required:\s*(.*?)$', jd_text, re.DOTALL | re.IGNORECASE)
    required_skills = []
    required_education = ""
    
    if required_section:
        req_text = required_section.group(1).strip()
        
        # Split by bullet points to get individual requirements
        requirements = re.split(r'[-•]\s*', req_text)
        
        for req in requirements:
            req = req.strip()
            if not req:
                continue
                
            # Check if it's education requirement
            if re.search(r"bachelor'?s|master'?s|degree|b\.|m\.", req, re.IGNORECASE):
                required_education = req
            elif not re.search(r'\d+\+?\s*years', req, re.IGNORECASE):
                # It's likely a skills requirement if it doesn't contain years
                # Handle comma-separated skills more carefully
                if ',' in req:
                    # Split by comma and clean up each skill
                    skills_in_line = [skill.strip() for skill in req.split(',') if skill.strip()]
                else:
                    # Split by spaces but preserve multi-word skills like "Adobe XD"
                    # Look for known multi-word skills first
                    multi_word_skills = ['Adobe XD', 'Adobe Photoshop', 'Adobe Illustrator', 'User Research', 'UX Research']
                    found_skills = []
                    remaining_text = req
                    
                    for multi_skill in multi_word_skills:
                        if multi_skill.lower() in remaining_text.lower():
                            found_skills.append(multi_skill)
                            remaining_text = remaining_text.replace(multi_skill, '', 1)
                    
                    # Add remaining single-word skills
                    single_skills = [skill.strip() for skill in remaining_text.split() if skill.strip() and len(skill.strip()) > 2]
                    skills_in_line = found_skills + single_skills
                
                required_skills.extend(skills_in_line)
    
    # Clean up required skills
    required_skills = [skill for skill in required_skills if skill and len(skill) > 1]
    
    return {
        'title': role,
        'required_experience_years': required_experience_years,
        'required_skills': required_skills,
        'required_education': required_education
    }

# Function to calculate semantic similarity with synonym/abbreviation handling
def semantic_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    
    # Normalize and expand common abbreviations/synonyms
    text1_normalized = normalize_skills_and_terms(str(text1))
    text2_normalized = normalize_skills_and_terms(str(text2))
    
    # Calculate semantic similarity
    emb1 = model.encode([text1_normalized])
    emb2 = model.encode([text2_normalized])
    similarity = util.cos_sim(emb1, emb2).item() * 100
    
    # Additional boost for known synonyms/abbreviations
    synonym_boost = calculate_synonym_match(text1.lower(), text2.lower())
    
    # Combine semantic similarity with synonym matching
    final_score = max(similarity, synonym_boost)
    
    return min(100, final_score)  # Cap at 100

# Function to normalize skills and expand abbreviations
def normalize_skills_and_terms(text):
    text = text.lower()
    
    # Common tech abbreviations and expansions
    abbreviations = {
        # Programming & Technologies
        'js': 'javascript',
        'ts': 'typescript', 
        'py': 'python',
        'html/css': 'html css web development',
        'css3': 'css',
        'html5': 'html',
        'react.js': 'react',
        'reactjs': 'react',
        'vue.js': 'vue',
        'vuejs': 'vue',
        'node.js': 'nodejs',
        'express.js': 'expressjs',
        
        # Design & UX
        'ui/ux': 'user interface user experience design',
        'ux/ui': 'user experience user interface design',
        'ux': 'user experience',
        'ui': 'user interface',
        'figma': 'figma design prototyping',
        'sketch': 'sketch design',
        'adobe xd': 'adobe xd design prototyping',
        'photoshop': 'adobe photoshop',
        'illustrator': 'adobe illustrator',
        
        # Data & Analytics
        'ml': 'machine learning',
        'ai': 'artificial intelligence',
        'nlp': 'natural language processing',
        'cv': 'computer vision',
        'dl': 'deep learning',
        
        # Methodologies
        'agile': 'agile methodology scrum',
        'scrum': 'agile scrum methodology',
        'devops': 'development operations',
        'ci/cd': 'continuous integration continuous deployment',
        
        # Databases
        'sql': 'structured query language database',
        'nosql': 'nosql database mongodb',
        'postgres': 'postgresql',
        'mysql': 'mysql database',
        
        # Cloud & Infrastructure
        'aws': 'amazon web services cloud',
        'gcp': 'google cloud platform',
        'azure': 'microsoft azure cloud',
        
        # Education
        'b.tech': 'bachelor technology engineering',
        'b.des': 'bachelor design',
        'm.tech': 'master technology engineering',
        'm.des': 'master design',
        'mba': 'master business administration',
        'ms': 'master science',
        'bs': 'bachelor science',
    }
    
    # Replace abbreviations with expanded forms
    for abbrev, expansion in abbreviations.items():
        if abbrev in text:
            text = text.replace(abbrev, expansion)
    
    return text

# Function to calculate direct synonym/abbreviation matches
def calculate_synonym_match(text1, text2):
    # Define synonym groups - skills that mean the same thing
    synonym_groups = [
        # Programming languages
        ['javascript', 'js', 'ecmascript'],
        ['typescript', 'ts'],
        ['python', 'py'],
        
        # Frontend frameworks
        ['react', 'react.js', 'reactjs'],
        ['vue', 'vue.js', 'vuejs'], 
        ['angular', 'angularjs'],
        
        # Design tools
        ['figma', 'figma design'],
        ['adobe xd', 'xd', 'adobe experience design'],
        ['sketch', 'sketch app'],
        ['photoshop', 'adobe photoshop', 'ps'],
        ['illustrator', 'adobe illustrator', 'ai'],
        
        # UX/UI terms
        ['ux', 'user experience', 'user experience design'],
        ['ui', 'user interface', 'user interface design'],
        ['ui/ux', 'ux/ui', 'user interface user experience'],
        ['prototyping', 'prototype', 'wireframing', 'wireframes'],
        ['user research', 'ux research', 'user studies'],
        
        # Data science
        ['machine learning', 'ml'],
        ['artificial intelligence', 'ai'],
        ['natural language processing', 'nlp'],
        ['computer vision', 'cv'],
        ['deep learning', 'dl'],
        
        # Cloud platforms
        ['aws', 'amazon web services'],
        ['gcp', 'google cloud platform', 'google cloud'],
        ['azure', 'microsoft azure'],
        
        # Databases
        ['postgresql', 'postgres'],
        ['mongodb', 'mongo'],
        
        # Methodologies
        ['agile', 'agile methodology'],
        ['scrum', 'scrum methodology'],
        ['ci/cd', 'continuous integration', 'continuous deployment'],
        
        # Education
        ['bachelor', "bachelor's", 'bachelors'],
        ['master', "master's", 'masters'],
        ['b.des', 'bachelor design', 'bachelor of design'],
        ['m.des', 'master design', 'master of design'],
        ['b.tech', 'bachelor technology', 'bachelor of technology'],
        ['m.tech', 'master technology', 'master of technology'],
    ]
    
    # Check if both texts contain terms from the same synonym group
    for group in synonym_groups:
        text1_matches = [term for term in group if term in text1]
        text2_matches = [term for term in group if term in text2]
        
        if text1_matches and text2_matches:
            return 95  # High match for synonyms
    
    return 0

# Function to calculate match scores with improved logic
def calculate_scores(resume_data, jd_data):
    # Experience Match: Compare years and semantic similarity of experience descriptions
    candidate_years = resume_data['experience_years']
    required_years = jd_data['required_experience_years']
    
    # Calculate years match (100% if candidate has required years or more)
    if required_years > 0:
        if candidate_years >= required_years:
            years_match = 100  # Perfect score if meets or exceeds requirements
            # Small bonus for exceeding (but cap at 100)
            if candidate_years > required_years:
                years_match = 100  # Keep at 100, no bonus needed
        else:
            years_match = (candidate_years / required_years) * 100
    else:
        years_match = 100 if candidate_years > 0 else 0
    
    # Semantic similarity of experience descriptions
    exp_description_matches = []
    if resume_data['experience'] and jd_data['title']:
        # Compare experience with job title and role relevance
        job_context = jd_data['title']
        for exp_line in resume_data['experience']:
            if exp_line:
                # Higher weight for role/title matching
                title_similarity = semantic_similarity(exp_line, job_context)
                exp_description_matches.append(title_similarity)
    
    exp_desc_score = np.mean(exp_description_matches) if exp_description_matches else 70  # Default good score if no description
    
    # Combine years and description match (80% years, 20% description for more weight on years)
    experience_match = (years_match * 0.8) + (exp_desc_score * 0.2)
    
    # Skills Match: Find best match for each required skill
    skills_matches = []
    if jd_data['required_skills'] and resume_data['skills']:
        for req_skill in jd_data['required_skills']:
            # Find the best matching skill from candidate's skills
            best_match = max([semantic_similarity(req_skill, candidate_skill) 
                            for candidate_skill in resume_data['skills']], default=0)
            skills_matches.append(best_match)
    
    skills_match = np.mean(skills_matches) if skills_matches else 0
    
    # Education Match: Semantic similarity with better preprocessing
    education_match = 0
    if resume_data['education'] and jd_data['required_education']:
        # Normalize education text for better matching
        candidate_edu = resume_data['education'].lower()
        required_edu = jd_data['required_education'].lower()
        
        # Enhanced keyword matching for degrees and fields
        degree_matches = 0
        field_matches = 0
        
        # Check for degree level
        degree_keywords = ['bachelor', 'master', 'b.des', 'm.des', 'b.tech', 'm.tech']
        required_degrees = ['bachelor', 'master']
        
        candidate_has_bachelor = any(keyword in candidate_edu for keyword in ['bachelor', 'b.des', 'b.tech', 'b.'])
        candidate_has_master = any(keyword in candidate_edu for keyword in ['master', 'm.des', 'm.tech', 'm.'])
        
        if any(req_deg in required_edu for req_deg in required_degrees):
            if candidate_has_bachelor or candidate_has_master:
                degree_matches = 1
        
        # Check for field relevance
        design_keywords = ['design', 'des', 'ui', 'ux', 'hci', 'human computer interaction', 'interaction']
        required_has_design = any(keyword in required_edu for keyword in design_keywords)
        candidate_has_design = any(keyword in candidate_edu for keyword in design_keywords)
        
        if required_has_design and candidate_has_design:
            field_matches = 1
        
        # Calculate education match
        if degree_matches and field_matches:
            education_match = 95  # Perfect match for degree level and field
        elif degree_matches:
            education_match = 75  # Good match for degree level
        elif field_matches:
            education_match = 60  # Partial match for field only
        else:
            # Fall back to semantic similarity
            education_match = semantic_similarity(resume_data['education'], jd_data['required_education'])
    
    # Overall Score: Weighted average (skills most important, then experience, then education)
    overall_score = (skills_match * 0.5) + (experience_match * 0.35) + (education_match * 0.15)
    
    return {
        'experience_match': round(experience_match),
        'skills_match': round(skills_match), 
        'education_match': round(education_match),
        'overall_score': round(overall_score)
    }

# POST endpoint for JSON response
@app.route('/match', methods=['POST'])
def match():
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file'}), 400
    resume_file = request.files['resume']
    jd_text = request.form.get('jd', '')

    # Save resume temporarily
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
                'overall_score': scores['overall_score'],
                'skills_match': scores['skills_match'],
                'experience_match': scores['experience_match'],
                'education_match': scores['education_match']
            }
        }
        return jsonify(response)
    finally:
        os.remove(resume_path)  # Clean up

# Bonus: Web UI routes
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
            'match_scores': scores
        })
    finally:
        os.remove(resume_path)

if __name__ == '__main__':
    app.run(debug=True)