# ResumeMatch
AI matching resumes against a job listing that outputs a graph that shows how similar each resume is to the job listing.

## Simple README for Resume Matching with Job Listing Code

### Purpose:
This script evaluates and compares a set of resumes against a given job listing. It assesses:
1. The semantic similarity between the job listing and each resume using embeddings.
2. The presence of certain required words or phrases in the resumes.

### Description:

1. **Import Libraries**: 
   - `pandas` for data handling.
   - `matplotlib` for plotting results.
   - `SentenceTransformer` for text embeddings.
   - `cosine_similarity` for calculating similarity between embeddings.

2. **Helper Functions**:
   - `read_file(file_name)`: Reads the content of a text file and returns it in lowercase.
   - `check_required_words(text, required_word_lists)`: Checks if a given text contains at least one word from each provided list of required words.

3. **Data Preparation**:
   - `job_listing` contains the filename of the job description text.
   - `resume_files` has the filenames of the resumes to be compared.
   - `required_word_lists` holds lists of words or phrases that are deemed important or necessary for the job.

4. **Reading Data**:
   - The job listing and all the resumes are read and stored.

5. **Text Embeddings**:
   - The `SentenceTransformer` model named 'all-MiniLM-L6-v2' is initialized. This model transforms textual data into embeddings (numeric representations).
   - The embeddings for both the job listing and the resumes are generated.

6. **Evaluation**:
   - Each resume's similarity score to the job listing is computed using cosine similarity.
   - The presence of required words or phrases in each resume is determined.
   - The results are stored in a list.

7. **Display Results**:
   - The results are converted into a pandas DataFrame and printed.
   - A bar and line plot is generated to visually represent the similarity scores and the presence of required words for each resume.

### How to Use:
1. Ensure the job listing and resume text files are in the appropriate directory.
2. Define any additional required words or phrases in the `required_word_lists`.
3. Run the script. The console will display a table of results, and a graphical plot will pop up showcasing the similarity scores and required word presence for each resume.
