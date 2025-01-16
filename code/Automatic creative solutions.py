import requests
import re
import json
import os


# Modified function to call Kimi API with added organization_id parameter
def call_kimi_api(prompt, api_key, title, project_id, organization_id, model="claude-3-5-sonnet-20240620", temperature=0.7):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Organization": organization_id,
        "OpenAI-Project": project_id,
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    try:
        response = requests.post("https://api.gptsapi.net/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        if 'choices' in response_json:
            return response_json['choices'][0]['message']['content']
        else:
            print("No 'choices' field found in the response.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling GPT API: {e}")
        return None


# Function to generate research ideas
def generate_idea(task_description, prev_ideas_string, api_key, title, project_id, organization_id):
    prompt = f"""
    You are an academic writing expert, proficient in writing rigorous, innovative, and clearly structured academic papers, and capable of providing valuable research ideas and clear writing guidance.
    '{task_description}' is the paper topic, and '{prev_ideas_string}' are preliminary prompts. '{title}' is the paper title queried from WOS based on the previous round's ideas. Discard any idea that is too similar to existing literature.
    Based on the above, propose the next research idea that has academic value, innovation, and feasibility.
    Respond in the following format, ensuring the use of English punctuation or standard punctuation, while writing the text in Chinese:
    Thought:
    <Answer>
    Generate Idea:
    ```json
    {{
        "Idea": "Short name of the idea",
        "Title": "Paper title",
        "Experiment Plan": "General structure or research method",
        "Clarity Score": "Provide a clarity score between 1 and 10",
        "Innovation Score": "Provide a research innovation score between 1 and 10",
        "Academic Value Score": "Provide an academic value score between 1 and 10",
        "Feasibility": "Provide a feasibility score between 1 and 10",
        "Data Requirement Match": "Provide a data requirement match score between 1 and 10",
        "Computational Complexity": "Provide a computational complexity score between 1 and 10",
        "Model Explainability": "Provide a model explainability score between 1 and 10",
        "Innovation Potential": "Provide an innovation potential score between 1 and 10",
        "Scalability": "Provide a scalability score between 1 and 10",
        "Experimental Validation Difficulty": "Provide an experimental validation difficulty score between 1 and 10",
        "Keywords": Write in English, maximum of 2 to 3 words, format ["keyword1", "keyword2"]
    }}
    ```json
    """
    response = call_kimi_api(prompt, api_key, title, project_id, organization_id)
    if response:
        return response
    else:
        print("Failed to generate research idea.")
        return None


# Modified reflection agent to provide feedback for improving ideas
def reflection_agent(doc_content, api_key, project_id, organization_id):
    prompt = f"""
    Idea Content:
    "{doc_content}"

    Based on the above idea, generate reflective feedback that includes the following:
    The main pros and cons of the idea. Provide suggestions for improvement. Only the body of the answer, no special symbols like ** or ##, or extra explanations.
    """

    reflection_feedback = call_kimi_api(prompt=prompt, api_key=api_key, title=None, project_id=project_id,
                                        organization_id=organization_id)

    return reflection_feedback


# Query Web of Science articles
def query_wos_articles(query, wos_api_key):
    RECORDS_TO_DISPLAY = 10
    url = f'https://api.clarivate.com/apis/wos-starter/v1/documents?q={query}&limit={RECORDS_TO_DISPLAY}&page=1&db=WOS&sortField=LD+D'
    headers = {'X-ApiKey': wos_api_key}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        articles = response.json().get('hits', [])
        return [{'title': article['title']} for article in articles]
    except requests.exceptions.RequestException as e:
        print(f"Error querying Web of Science: {e}")
        return []


# Extract keywords and check innovation in Web of Science
def extract_keywords_and_check_innovation(new_idea, wos_api_key):
    try:
        new_idea_json_match = re.search(r'\n```json\n(.*?)\n```', new_idea, re.DOTALL)
        if new_idea_json_match:
            new_idea_json_str = new_idea_json_match.group(1)
            new_idea_json = json.loads(new_idea_json_str)
            keywords = new_idea_json.get("Keywords", [])

            print("Extracted keywords:", keywords)

            search_query = f'TI={" ".join(keywords)}'
            print("Constructed search query:", search_query)
            wos_articles = query_wos_articles(search_query, wos_api_key)

            print("Returned articles from Web of Science:", wos_articles)

            included_titles = []
            partially_included_titles = []
            excluded_titles = []

            for article in wos_articles:
                title = article['title']
                if all(keyword.lower() in title.lower() for keyword in keywords):
                    included_titles.append(title)
                elif any(keyword.lower() in title.lower() for keyword in keywords):
                    partially_included_titles.append(title)
                else:
                    excluded_titles.append(title)

            print("\nArticles that include all keywords:")
            for title in included_titles:
                print(" -", title)

            print("\nArticles that partially include keywords:")
            for title in partially_included_titles:
                print(" -", title)

            print("\nArticles that do not include any keywords:")
            for title in excluded_titles:
                print(" -", title)

            if len(included_titles) >= 7:
                print("Innovation insufficient, regenerating idea...")
                return False, included_titles
            print("Sufficient innovation")
            return True, included_titles
        else:
            print("No 'new idea:' format content found.")
            return False, []

    except Exception as e:
        print(f"Error extracting keywords or checking innovation: {e}")
        return False, []


# Extract Experiment Plan section
def extract_experiment_from_idea(new_idea):
    try:
        json_match = re.search(r'\s*```json\s*(\{.*?\})\s*```', new_idea, re.DOTALL)
        if json_match:
            new_idea_json_str = json_match.group(1).strip()
            new_idea_json = json.loads(new_idea_json_str)
            return new_idea_json.get("Experiment Plan", None)
        else:
            print("No 'Generate Idea' section found.")
            return None
    except json.JSONDecodeError as e:
        print(f"Error extracting Experiment Plan: {e}")
        return None


# Main function to handle multiple rounds of idea generation and reflection
def iterative_process(task_description, prev_ideas_string, num_reflections, api_key, wos_api_key, project_id, organization_id):
    current_round = 1
    experiment_description = None
    last_thought = ""
    title = None

    for _ in range(num_reflections):
        print(f"\n--- Round {current_round} ---")

        if last_thought:
            prev_ideas_string += f"\nPrevious Round Reflection: {last_thought}"

        # Generate idea
        new_idea = generate_idea(task_description, prev_ideas_string, api_key, title, project_id, organization_id)
        if new_idea:
            print(f"Generated Idea: {new_idea}")

            # Generate reflection feedback
            last_thought = reflection_agent(new_idea, api_key, project_id, organization_id)
            if last_thought:
                print(f"Reflection Feedback: {last_thought}")
            else:
                last_thought = "Reflection generation failed, unable to provide further suggestions."

            # Check innovation
            is_innovative, included_titles = extract_keywords_and_check_innovation(new_idea, wos_api_key)
            title = included_titles  # Pass query results as title

            # Extract experiment description on the third round
            if current_round == 3:
                experiment_description = extract_experiment_from_idea(new_idea)
                if experiment_description:
                    print(f"Experiment Description from Round 3: {experiment_description}")
                    break
        else:
            print("Failed to generate new idea.")
            break

        # Increment round count regardless of innovation check result
        current_round += 1

    return experiment_description


if __name__ == "__main__":
    task_description = "Write a paper on the automated idea generation and machine learning validation framework for predicting dielectric constants in electrolyte solutions."
    prev_ideas_string = "Attempt using survival analysis algorithms to improve model accuracy."
    api_key = ""  # Replace with your actual API key
    wos_api_key = ""  # Replace with your Web of Science API key
    organization_id = ""  # Replace with your actual organization ID
    project_id = "t1"  # Replace with your actual project ID
experiment_description = iterative_process(task_description, prev_ideas_string, 3, api_key, wos_api_key, project_id, organization_id)
