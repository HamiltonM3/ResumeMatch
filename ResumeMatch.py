import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(file_name):
    """Read file and return its text content."""
    with open(file_name, 'r') as file:
        return file.read().lower()

def check_required_words(text, required_word_lists):
    """Check if at least one word from each list of required words is present in the text."""
    text_lower = text.lower()  # convert text to lowercase
    for word_list in required_word_lists:
        word_list_lower = [word.lower() for word in word_list]  # convert required words to lowercase
        if not any(word in text_lower for word in word_list_lower):
            return False  # if no word from a list is found, return False
    return True  # if one work in all the lists are found, return True

# file name of the job listing
job_listing = 'job_listing.txt'

# file name of the resumes
resume_files = ['resume1.txt', 'resume2.txt', 'resume3.txt', 'resume4.txt']  # add more if needed

# get list of all text files in the directory instead of individual files
# resume_files = [os.path.join(resume_dir, file) for file in os.listdir(resume_dir) if file.endswith('.txt')]


# list of required words
required_word_lists = [['python', 'java', 'c++', 'bash'], ['Security+', 'SEC+', 'CEH', 'CISSP', 'CASP', 'CompTIA Advanced Security Practitioner', 'CASP+', 'Certified Ethical Hacker', 'Certified Information Systems Security Professional'], [
'ba', 'bs', 'ma', 'ms', 'mba', 'bachelors', 'masters']]  # add more if needed

# read job listing
job_text = read_file(job_listing)

# read resumes
resume_texts = [read_file(resume) for resume in resume_files]

# all documents
documents = [job_text] + resume_texts

# initialize the model using NLP BERT's all-MiniLM-L6-v2
model = SentenceTransformer('all-MiniLM-L6-v2')

# generate embeddings
embeddings = model.encode(documents)

# list to store results
results_list = []

# calculate similarity score 
for i, resume in enumerate(resume_files):
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[i+1]])
    required_words_present = check_required_words(resume_texts[i], required_word_lists)
    results_list.append({'Resume': resume, 'Similarity Score': similarity_score[0][0], 'Required Words Present': required_words_present})

# convert list to DataFrame
results = pd.DataFrame(results_list)

print(results)

# knowledge representation plot
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Resume')
ax1.set_ylabel('Similarity Score', color=color)
ax1.bar(results['Resume'], results['Similarity Score'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Required Words Present', color=color)
ax2.plot(results['Resume'], results['Required Words Present'], color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
