import requests
import matplotlib.pyplot as plt

# Replace with your GitLab project info
GITLAB_URL = 'https://gitlab.com'  # GitLab instance URL
PRIVATE_TOKEN = 'glpat-edJX_2U4rcnr_dmVxoFM'  # GitLab private access token
NAME_SPACE = 'chazza1414'
PROJECT_ID = 'lip-reader'  # GitLab project ID or URL-encoded project path (e.g., 'namespace/project')

# GitLab API URL for commits
API_URL = f"{GITLAB_URL}/api/v4/projects/{NAME_SPACE}/{PROJECT_ID}/repository/commits"

print(API_URL)

# Function to fetch commits from GitLab
def fetch_commits(api_url, private_token, per_page=100):
    commits = []
    page = 1
    
    while True:
        # Prepare headers for authentication
        headers = {
            'PRIVATE-TOKEN': private_token
        }
        
        # Request the commit data
        response = requests.get(api_url, headers=headers, params={'page': page, 'per_page': per_page})
        
        if response.status_code != 200:
            print(f"Error fetching commits: {response.status_code}")
            print(response)
            break
        
        data = response.json()
        if not data:
            break
        
        commits.extend(data)
        page += 1
        
    return commits

# Function to extract line changes (additions and deletions)
def extract_line_changes(commits):
    additions = []
    deletions = []
    
    for commit in commits:
        # Extract stats for each commit
        for diff in commit['stats']:
            additions.append(diff['additions'])
            deletions.append(diff['deletions'])
    
    return additions, deletions

# Fetch commits from GitLab
commits = fetch_commits(API_URL, PRIVATE_TOKEN)

# Extract line changes (additions and deletions)
additions, deletions = extract_line_changes(commits)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(additions, label='Additions', color='g', marker='o')
plt.plot(deletions, label='Deletions', color='r', marker='x')
plt.xlabel('Commit Index')
plt.ylabel('Lines Changed')
plt.title('Line Changes per Commit')
plt.legend()
plt.grid(True)
plt.show()
