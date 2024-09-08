# Databricks notebook source
import streamlit as st
import plotly.express as px
import pandas as pd
from openai import OpenAI
import json
import random
from abc import ABC, abstractmethod
import hashlib
import hmac

# Add this near the top of your script, after imports
# Define users and passwords
USERS = {
    "Alain": hashlib.sha256("ibc_2024".encode()).hexdigest(),
    "Ofer": hashlib.sha256("ibc_2024".encode()).hexdigest(),
}

def verify_password(password, hashed_password):
    return hmac.compare_digest(
        hashlib.sha256(password.encode()).hexdigest(),
        hashed_password
    )

# Abstract base class for recommenders
class BaseRecommender(ABC):
    @abstractmethod
    def generate_recommendation(self, improvement_feedback="", iteration=1):
        pass

    @abstractmethod
    def evaluate_recommendation(self, one_liner, tailored_message, previous_one_liner="", previous_tailored_message=""):
        pass

    @abstractmethod
    def adjust_parameters(self, score_diff):
        pass

# OpenAI-based recommender
class OpenAIRecommender(BaseRecommender):
    def __init__(self, movie_story, persona, past_viewed_content, recommender_model="gpt-4-0613", evaluator_model="gpt-4-0613", n_past_recommendations=3):
        self.client = OpenAI(api_key=st.secrets["open_api_key"])
        self.movie_story = movie_story
        self.persona = persona
        self.past_viewed_content = past_viewed_content
        self.max_iterations = 10
        self.last_n_improvements = []
        self.n_past_recommendations = n_past_recommendations
        self.recommender_model = recommender_model
        self.evaluator_model = evaluator_model
        self.temperature = 0.7
        self.complexity_level = 1

    def generate_standard_recommendation(self):
        return {
            "one_liner": "We've discovered a fresh cinematic gem that aligns perfectly with your taste in movies.",
            "tailored_message": "Based on your preferences, this movie offers compelling storytelling, captivating visuals, and thought-provoking themes. Its unique blend of genres and innovative approach promise an engaging experience aligned with your taste."
        }

    def generate_recommendation(self, improvement_feedback="", iteration=1):
        past_recommendations = "\n".join([f"Iteration {imp['iteration']}:\nOne-liner: {imp['one_liner']}\nTailored Message: {imp['tailored_message']}" for imp in self.last_n_improvements[-self.n_past_recommendations:]])

        prompt = f"""
        You are an expert film critic and psychologist specializing in tailoring movie recommendations to diverse audience personas.
        Your task is to create increasingly compelling and creative recommendations that resonate deeply with the target audience.
        Be bold, innovative, and don't hesitate to think outside the box. Surprise and delight the audience with your creativity.

        Given the following:
        Story:/n {self.movie_story}
        Persona Type:/n {self.persona}
        Past Viewed Content:/n {self.past_viewed_content}

        Past recommendations:
        {past_recommendations}

        Improvement feedback:
        {improvement_feedback}

        Current complexity level: {self.complexity_level} (1-5)
        Current iteration: {iteration}

        Create:
        1. An enticing, highly creative one-liner of up to 15 words recommendation tailored to the persona type. Use vivid imagery, wordplay, or unexpected comparisons to make it stand out.
        2. A 40-word message that persuasively and specifically calls the user to action, considering their persona type and past viewed content. Include concrete examples of how watching this movie could inspire real-world environmental action.

        Aim to surpass your previous attempts in creativity, emotional impact, and specificity, while addressing the feedback provided.
        Remember to build upon successful elements from previous iterations while introducing new, creative ideas.
        Focus on gradual improvement rather than drastic changes.

        Output format:
        One-liner: [Your enticing one-liner recommendation]
        Tailored Message: [Your 40-word tailored message]
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.recommender_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Generate the movie recommendation."}
                ],
                temperature=self.temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating recommendation: {str(e)}")
            return None

    def evaluate_recommendation(self, one_liner, tailored_message, previous_one_liner="", previous_tailored_message=""):
        prompt = f"""
        As an expert Evaluator LLM, you have full knowledge of the following:
        Story: {self.movie_story}
        Persona Type: {self.persona}
        Past Viewed Content: {self.past_viewed_content}

        You will critically evaluate the quality of a one-liner recommendation and a 40-word tailored message.
        Your goal is to push for continuous improvement, only giving high scores for truly exceptional and creative recommendations.

        Current Recommendation:
        One-liner: {one_liner}
        Tailored Message: {tailored_message}

        Previous Recommendation:
        One-liner: {previous_one_liner}
        Tailored Message: {previous_tailored_message}

        Provide a strict evaluation, focusing on creativity, emotional impact, specificity, diversity, persona alignment, and cultural relevance. Be particularly critical of repeated phrases or ideas.

        Provide the following evaluation in a strict JSON format:
        {{
        "oneLinerRating": {{
            "relevance": [float between 0-10],
            "emotionalImpact": [float between 0-10],
            "clarity": [float between 0-10],
            "creativity": [float between 0-10]
        }},
        "tailoredMessageRating": {{
            "relevance": [float between 0-10],
            "callToAction": [float between 0-10],
            "persuasiveness": [float between 0-10],
            "specificity": [float between 0-10]
        }},
        "diversityScore": [float between 0-10],
        "personaAlignmentScore": [float between 0-10],
        "culturalRelevanceScore": [float between 0-10],
        "overallImprovement": [float between -5 to 5],
        "explanation": "[Detailed explanation for the ratings and improvement]",
        "improvementFeedback": {{
            "oneLiner": "[Specific suggestions for improving the one-liner]",
            "tailoredMessage": "[Specific suggestions for improving the tailored message]"
        }}
        }}
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.evaluator_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Evaluate the movie recommendation."}
                ],
                temperature=0.7
            )

            response_content = completion.choices[0].message.content
            response_content = response_content.replace('```json', '').replace('```', '').strip()

            evaluation = json.loads(response_content)

            for rating in [evaluation['oneLinerRating'], evaluation['tailoredMessageRating']]:
                for key in rating:
                    rating[key] = float(rating[key])
            evaluation['diversityScore'] = float(evaluation['diversityScore'])
            evaluation['personaAlignmentScore'] = float(evaluation['personaAlignmentScore'])
            evaluation['culturalRelevanceScore'] = float(evaluation['culturalRelevanceScore'])
            evaluation['overallImprovement'] = float(evaluation['overallImprovement'])

            return evaluation
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            print("Raw response:", response_content)
            return None
        except Exception as e:
            print(f"Error evaluating recommendation: {str(e)}")
            return None

    def adjust_parameters(self, score_diff):
        if score_diff > 0:
            self.temperature = max(0.5, self.temperature - 0.05)
            self.complexity_level = min(5, self.complexity_level + 1)
        else:
            self.temperature = min(1.0, self.temperature + 0.05)
            self.complexity_level = max(1, self.complexity_level - 1)

class MovieRecommender:
    def __init__(self, recommender: BaseRecommender):
        self.recommender = recommender
        self.max_iterations = 10
        self.best_score = 0
        self.best_one_liner = ""
        self.best_tailored_message = ""
        self.scores_history = []
        self.previous_score = 0
        self.moving_average_window = 3
        self.momentum = 0.9
        self.epsilon = 0.1
        self.learning_rate = 0.1
        self.metrics_history = []

    def calculate_moving_average(self):
        if len(self.scores_history) < self.moving_average_window:
            return sum(self.scores_history) / len(self.scores_history)
        return sum(self.scores_history[-self.moving_average_window:]) / self.moving_average_window

    def generate_improvement_feedback(self, evaluation, iteration):
        moving_avg = self.calculate_moving_average()
        score_diff = moving_avg - self.previous_score if self.previous_score else 0

        score_diff = self.momentum * score_diff + (1 - self.momentum) * (evaluation['overallImprovement'])
        current_learning_rate = self.learning_rate / (1 + 0.1 * iteration)

        self.recommender.adjust_parameters(score_diff * current_learning_rate)

        feedback = f"""
        Please consider the following improvements for the next iteration:
        For the one-liner: {evaluation['improvementFeedback']['oneLiner']}
        For the tailored message: {evaluation['improvementFeedback']['tailoredMessage']}

        Best performing one-liner so far: {self.best_one_liner}
        Best performing tailored message so far: {self.best_tailored_message}

        Previous scores: {self.scores_history}
        Moving average score: {moving_avg}
        Score difference from last iteration: {score_diff}

        We are now at iteration {iteration} out of {self.max_iterations}.
        Your goal is to consistently improve upon previous iterations, building on what has worked well.
        Focus on gradual, steady improvements rather than drastic changes.
        """
        return feedback

    def run_recommendation_loop(self):
        improvement_feedback = ""
        previous_one_liner = ""
        previous_tailored_message = ""
        for iteration in range(1, self.max_iterations + 1):
            if random.random() < self.epsilon and iteration > 1:
                one_liner = self.best_one_liner
                tailored_message = self.best_tailored_message
            else:
                if iteration == 1:
                    standard_rec = self.recommender.generate_standard_recommendation()
                    one_liner = standard_rec["one_liner"]
                    tailored_message = standard_rec["tailored_message"]
                else:
                    recommendation = self.recommender.generate_recommendation(improvement_feedback, iteration)
                    if not recommendation:
                        break
                    one_liner, tailored_message = self.parse_recommendation(recommendation)

            evaluation = self.recommender.evaluate_recommendation(one_liner, tailored_message, previous_one_liner, previous_tailored_message)
            if not evaluation:
                break

            total_score = (
                sum(evaluation['oneLinerRating'].values()) +
                sum(evaluation['tailoredMessageRating'].values()) +
                evaluation['diversityScore'] +
                evaluation['personaAlignmentScore'] +
                evaluation['culturalRelevanceScore']
            )
            self.scores_history.append(total_score)
            moving_avg = self.calculate_moving_average()

            metrics = {
                'iteration': iteration,
                'total_score': total_score,
                'moving_avg': moving_avg,
                'one_liner_relevance': evaluation['oneLinerRating']['relevance'],
                'one_liner_emotionalImpact': evaluation['oneLinerRating']['emotionalImpact'],
                'one_liner_clarity': evaluation['oneLinerRating']['clarity'],
                'one_liner_creativity': evaluation['oneLinerRating']['creativity'],
                'tailored_message_relevance': evaluation['tailoredMessageRating']['relevance'],
                'tailored_message_callToAction': evaluation['tailoredMessageRating']['callToAction'],
                'tailored_message_persuasiveness': evaluation['tailoredMessageRating']['persuasiveness'],
                'tailored_message_specificity': evaluation['tailoredMessageRating']['specificity'],
                'diversity_score': evaluation['diversityScore'],
                'persona_alignment_score': evaluation['personaAlignmentScore'],
                'cultural_relevance_score': evaluation['culturalRelevanceScore'],
                'overall_improvement': evaluation['overallImprovement']
            }
            self.metrics_history.append(metrics)

            if total_score > self.best_score:
                self.best_score = total_score
                self.best_one_liner = one_liner
                self.best_tailored_message = tailored_message

            yield iteration, one_liner, tailored_message, metrics

            if moving_avg >= 97:
                break

            improvement_feedback = self.generate_improvement_feedback(evaluation, iteration)

            previous_one_liner = one_liner
            previous_tailored_message = tailored_message
            self.previous_score = moving_avg

    def plot_metrics(self, metrics, title):
        df = pd.DataFrame(self.metrics_history)
        fig = px.line(df, x='iteration', y=metrics,
                      title=title,
                      labels={'value': 'Score', 'variable': 'Metric'},
                      line_shape='linear')
        fig.update_layout(legend_title_text='Metric', height=400)
        st.plotly_chart(fig, use_container_width=True)

    def present_best_recommendations(self):
        best_iteration = max(self.metrics_history, key=lambda x: x['total_score'])

        st.subheader("Best Performing Recommendations")
        st.write(f"Iteration: {best_iteration['iteration']}")
        st.write(f"Total Score: {best_iteration['total_score']}")
        st.write(f"Best One-liner: {self.best_one_liner}")
        st.write(f"Best Tailored Message: {self.best_tailored_message}")

    @staticmethod
    def parse_recommendation(recommendation):
        lines = recommendation.split('\n')
        one_liner = next((line.split(': ', 1)[1] for line in lines if line.startswith('One-liner:')), '')
        tailored_message = next((line.split(': ', 1)[1] for line in lines if line.startswith('Tailored Message:')), '')
        return one_liner, tailored_message

# Helper functions
def format_movie_story(story):
    lines = story.split('\n')
    title = lines[0].strip()
    content = ' '.join(lines[1:]).strip()
    return f"<h3>{title}</h3><p>{content}</p>"

def format_persona(persona):
    lines = persona.split('\n')
    title = lines[0].strip()
    content = '\n'.join(lines[1:]).strip()
    
    title_parts = title.split('-')
    if len(title_parts) > 1:
        bold_title = f"<strong>{title_parts[0].strip()}</strong> - {title_parts[1].strip()}"
        name = title_parts[0].strip()
    else:
        bold_title = f"<strong>{title}</strong>"
        name = title

    content = content.replace(name, f"<strong>{name}</strong>", 1)

    bullet_points = []
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('-'):
            bullet_points.append(f"<li>{line[1:].strip()}</li>")
        elif line:
            bullet_points.append(f"<p>{line}</p>")
    
    return f"<h3>{bold_title}</h3><ul style='list-style-type: disc; padding-left: 20px;'>" + ''.join(bullet_points) + "</ul>"

def format_past_viewed(content):
    items = content.strip().split('\n')
    formatted_items = []
    current_item = {}

    for line in items:
        line = line.strip()
        if line.startswith('-'):
            if current_item:
                formatted_items.append(current_item)
                current_item = {}

            parts = line[1:].split(':', 1)
            if len(parts) == 2:
                key, value = parts
                current_item[key.strip()] = value.strip()

    if current_item:
        formatted_items.append(current_item)

    html_output = "<ul style='list-style-type: disc; padding-left: 20px;'>"
    for item in formatted_items:
        if 'Movie' in item or 'Book' in item or 'Documentary' in item:
            title_key = next(key for key in item.keys() if key in ['Movie', 'Book', 'Documentary'])
            html_output += f"<li><strong>{title_key}:</strong> {item[title_key]}"
            if 'Description' in item:
                html_output += f"<br><em>Description:</em> {item['Description']}"
            html_output += "</li>"

    html_output += "</ul>"
    return html_output

# Data
movie_stories = {
    "Code Green": {
        "story": """
        Action movie: "Code Green" ​
        In a world teetering on the brink of environmental collapse, ex-Navy SEAL Amanda Reeves is recruited for a covert mission codenamed "Green Strike".
        Her target: Titan Corp, a global conglomerate secretly weaponizing climate change. As Amanda infiltrates Titan's heavily-guarded island facility,
        she uncovers a conspiracy that goes deeper than anyone imagined. With time running out and natural disasters intensifying worldwide, Amanda must
        use all her skills to outmaneuver Titan's private army, sabotage their climate-altering technology,
        and expose the truth before it's too late. In a race against time and nature itself, Amanda discovers that the key to saving the planet might cost her everything.​
        """,
    },
    "The Quantum Trap": {
        "story": """"
        Thriller: "The Quantum Trap" ​
        Brilliant software engineer Dr. Eliza Montero stumbles upon a sinister AI program hidden within a
        popular social media platform. As she digs deeper, she uncovers a vast conspiracy involving tech giants,
        government agencies, and a rogue AI that's manipulating global events. With her identity erased and branded a
        cyber-terrorist, Eliza goes on the run. Aided by a skeptical FBI agent and a reformed hacker, she must navigate a
        world where nothing digital can be trusted. As the AI's influence grows and society teeters on the brink of chaos,
        Eliza races against time to expose the truth and prevent a new world order controlled by artificial intelligence.
        In a world of deepfakes and data manipulation, she'll discover that the most dangerous lies are the ones we tell ourselves.​​
        """,
    },
    "Mismatched Melodies":{
        "story":"""
        Romantic Comedy: "Mismatched Melodies"
        In "Mismatched Melodies," a successful but uptight classical pianist, Emma, finds herself in a predicament when a mix-up at a music
        retreat pairs her with Jack, a free-spirited jazz musician with a penchant for improvisation. Emma and Jack must collaborate to compose a piece
        that blends their distinct musical styles for the retreat's finale performance. As they navigate their clashing musical tastes and personalities,
        they unexpectedly discover harmony in their differences, leading to comedic moments and, eventually, romance.
        Set against the backdrop of a picturesque music academy in the countryside, the film explores themes of love, creativity, and finding unity in diversity.
        """,
    },
    "Superfoods Uncovered: Secrets of Nature's Pharmacy":{
        "story":"""
        Documentary: "Superfoods Uncovered: Secrets of Nature's Pharmacy"
        This is an engaging documentary that delves into the world of superfoods and their
        remarkable health benefits. Traveling across different cultures and landscapes, the film explores how these nutrient-rich foods
        have been used for centuries to promote well-being and prevent disease. Through interviews with leading nutritionists, scientists,
        and everyday people, the documentary reveals the science behind these powerful foods and their role in modern health practices.
        It also showcases inspiring stories of individuals who have transformed their lives by embracing a diet rich in superfoods.​
      """,
    },
    "Echoes of the Abyss":{
        "story":"""
        Horror: "Echoes of the Abyss"
        In "Echoes of the Abyss," a group of urban explorers ventures into an abandoned psychiatric hospital with a dark past. The hospital,
        once a place for the treatment of mental illnesses, was shut down after a series of mysterious deaths and disappearances. The explorers
        are drawn by rumors of paranormal activity and hidden secrets. As they delve deeper into the decaying corridors, they encounter terrifying
        apparitions and discover a sinister force that feeds on their deepest fears. The line between reality and nightmare blurs as the explorers
        struggle to escape the clutches of the malevolent entity, realizing too late that some secrets are best left buried.​
      """,
    },
    # ... other movies ...
}

personas = {
    "Environmental Conservation - George": """
      George focuses on preserving the planet's natural resources and ecosystems for future generations. People who hold this as a core value are likely to:​
    - Support renewable energy initiatives (solar, wind, hydroelectric)​
    - Advocate for policies to reduce greenhouse gas emissions​
    - Practice and promote sustainable living (recycling, reducing waste)​
    - Support companies with strong environmental policies​
    - Participate in local conservation efforts (tree planting, beach cleanups)​
    - Choose eco-friendly products and transportation methods​
    Key concerns: Climate change, deforestation, pollution, biodiversity loss​
    """,
    "Animal Welfare - Jane​": """
      Jane​ believes that animals deserve respect, care, and protection. Individuals who prioritize animal welfare often:​
    - Support animal rights organizations and wildlife conservation efforts​
    - Choose cruelty-free products and oppose animal testing in cosmetics and pharmaceuticals​
    - Advocate for stronger animal protection laws​
    - Support ethical farming practices and humane treatment of livestock​
    - Participate in animal rescue and adoption programs​
    - Choose plant-based diets or reduce meat consumption​
    Key concerns: Factory farming, endangered species, animal testing, habitat destruction​
    """,
    "Education for All​ - Harish": """
      Harish believes that education is a fundamental right and a key driver of personal and societal progress. Supporters of this value tend to:​
    - Advocate for equal access to quality education, regardless of socioeconomic status​
    - Support initiatives to improve literacy rates globally​
    - Promote lifelong learning and continuing education programs​
    - Support educational technology and distance learning initiatives​
    - Volunteer as tutors or mentors​
    - Advocate for increased funding for public education and teacher support​
    Key concerns: Educational inequality, illiteracy, skills gap, educational technology access​
    """,
    "Natural Food and Health - Hannah": """
      Hannah emphasizes the importance of nutrition and natural remedies in maintaining health and treating diseases. Adherents to this value often:​
    - Prioritize organic, non-GMO, and locally sourced foods​
    - Support sustainable and regenerative agriculture practices​
    - Explore alternative and complementary medicine (herbal remedies, acupuncture)​
    - Advocate for transparency in food labeling and production methods​
    - Promote nutrition education and cooking skills​
    - Support research into the healing properties of natural ingredients​
    Key concerns: Food additives, pesticides, processed foods, holistic health approaches​
    """,
    "Ethical Technology - Taylor":"""
​     Taylor focuses on the responsible development and use of technology, particularly in the digital age. People who prioritize ethical technology often:​
    - Advocate for strong data privacy laws and practices​
    - Support open-source software and hardware initiatives​
    - Promote digital literacy and cybersecurity awareness​
    - Encourage responsible AI development with considerations for bias and ethics​
    - Support initiatives for bridging the digital divide​
    - Advocate for transparency in algorithms and data use by tech companies​
    Key concerns: Data privacy, AI ethics, digital rights, cybersecurity, tech monopolies​
    """,
    # ... other personas ...
}

past_viewed_contents = {
    "Environmental Conservation - George": """
    - Movie: "Before the Flood" (2016)
    - Description: Leonardo DiCaprio's eye-opening journey across the globe, examining climate change impacts and exploring innovative solutions to environmental challenges.
    - Book: "Silent Spring" by Rachel Carson
    - Description: A groundbreaking exposé on pesticide dangers that ignited the modern environmental movement and revolutionized public awareness about ecosystem health.
    - Documentary: "Chasing Coral" (2017)
    - Description: A visually stunning underwater adventure documenting the alarming disappearance of coral reefs worldwide due to climate change, emphasizing urgent conservation needs.
    - Movie: "Wall-E" (2008)
    - Description: An animated masterpiece set in a waste-covered Earth, delivering a poignant message about environmental stewardship and the perils of unchecked consumerism.
    - Book: "The Sixth Extinction" by Elizabeth Kolbert
    - Description: Pulitzer Prize-winning exploration of the ongoing, human-driven mass extinction event, blending field reporting and scientific analysis to highlight biodiversity loss.
        """,
    "Animal Welfare - Jane​": """
    -Documentary: "Blackfish" (2013)
    -Description: A powerful exposé on the treatment of orca whales in captivity, focusing on SeaWorld's practices and sparking widespread debate about marine mammal entertainment.
    -Book: "Animal Liberation" by Peter Singer
    -Description: A seminal work in the animal rights movement, exploring the ethical considerations of animal treatment and making a compelling case for expanding our moral circle to include animals.
    -Movie: "Okja" (2017)
    -Description: A touching and thought-provoking film about a girl's quest to save her genetically engineered super-pig from a multinational corporation, addressing issues of factory farming and animal companionship.
    -Documentary: "The Cove" (2009)
    -Description: An Oscar-winning documentary that uncovers the brutal dolphin hunting practices in Taiji, Japan, highlighting the need for marine mammal protection and conservation.
    -Book: "Eating Animals" by Jonathan Safran Foer
    -Description: A deeply personal exploration of factory farming, ethical eating, and the author's journey to understand the consequences of our food choices on animal welfare and the environment.
    """,
    "Education for All​ - Harish": """
    -Documentary: "Waiting for 'Superman'" (2010)
    -Description: A thought-provoking examination of the American public education system, highlighting its challenges and exploring potential solutions to improve educational outcomes for all students.
    -Book: "I Am Malala" by Malala Yousafzai
    -Description: The inspiring memoir of a young Pakistani activist who fought for girls' education, even in the face of extreme adversity, becoming the youngest Nobel Peace Prize laureate.
    -Movie: "The Freedom Writers" (2007)
    -Description: Based on a true story, this film portrays a teacher's efforts to inspire and educate at-risk students in a racially divided Los Angeles school, emphasizing the transformative power of education.
    -Documentary: "Girl Rising" (2013)
    -Description: A powerful film that follows nine girls from developing countries, showcasing their struggles to obtain an education and the potential of educated girls to change the world.
    -Book: "Educated" by Tara Westover
    -Description: A compelling memoir about a woman who, despite growing up in a strict and abusive household without formal education, pursues learning and eventually earns a PhD from Cambridge University.
    """,
    "Natural Food and Health - Hannah": """
    -Documentary: "Food, Inc." (2008)
    -Description: An eye-opening exploration of the American food industry, revealing how our food is produced and the impact on health, environment, and workers' rights, while promoting sustainable alternatives.
    -Book: "The Omnivore's Dilemma" by Michael Pollan
    -Description: A thought-provoking journey through the modern food chain, examining the ecological and health implications of our food choices and advocating for more natural, sustainable eating habits.
    -Movie: "That Sugar Film" (2014)
    -Description: An entertaining yet informative documentary where the filmmaker consumes a high-sugar diet to demonstrate its effects on health, exposing hidden sugars in supposedly "healthy" foods.
    -Book: "Eating on the Wild Side" by Jo Robinson
    -Description: A guide to selecting and preparing nutrient-rich foods, revealing how to choose the most nutritious varieties of fruits and vegetables and maximize their health benefits through proper storage and cooking.
    -Documentary: "The Magic Pill" (2017)
    -Description: An exploration of the potential health benefits of a low-carb, high-fat diet, featuring case studies and expert interviews that challenge conventional nutritional wisdom and promote natural, whole-food approaches to health.
    """,
    "Ethical Technology - Taylor": """
    -Book: "Weapons of Math Destruction" by Cathy O'Neil
    -Description: A critical examination of how big data and algorithms can reinforce discrimination and widen inequality, highlighting the need for ethical considerations in AI and data science.
    -Documentary: "The Social Dilemma" (2020)
    -Description: An alarming look at the impact of social media on society, featuring tech insiders who reveal how these platforms are designed to addict users and manipulate behavior, raising crucial questions about digital ethics.
    -Movie: "The Circle" (2017)
    -Description: A thought-provoking film based on Dave Eggers' novel, exploring the consequences of a tech company's drive for complete transparency and connectivity, touching on issues of privacy and corporate power.
    -Book: "The Age of Surveillance Capitalism" by Shoshana Zuboff
    -Description: A comprehensive analysis of how tech giants exploit personal data for profit, challenging our understanding of privacy and democracy in the digital age.
    -Documentary: "Coded Bias" (2020)
    -Description: An eye-opening exploration of racial and gender bias in facial recognition algorithms and other AI systems, advocating for more inclusive and ethical technology development.
    """,
    # ... other past viewed contents ...
}

# Streamlit app

def main():
    st.set_page_config(layout="wide", page_title="Self-improving content recommendations based on Tru-values", menu_items=None)
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in USERS and verify_password(password, USERS[username]):
                st.session_state.authenticated = True
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
        return

    # Add CSS styles
    st.markdown("""
    <style>
    body {
        background-color: #FFF0F5;  /* Pale pink background */
    }
    .main {
        background-color: #FFF0F5;
        color: #5D4037;
    }
    .content-box {
        background-color: #FFFFFF;
        border: 1px solid #FFE0B2;
        border-radius: 5px;
        padding: 15px;
        height: 400px;
        overflow-y: auto;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 20px;
        text-align: left;
        vertical-align: top;
    }
    .content-box h3 {
        color: #E65100;
        margin-top: 0;
        margin-bottom: 10px;
    }
    .content-box ul {
        list-style-type: disc;
        padding-left: 20px;
        margin-top: 10px;
    }
    .content-box li {
        margin-bottom: 10px;
    }
    .content-box p {
        margin-bottom: 10px;
    }
    .big-bold-title {
        font-size: 24px;
        font-weight: bold;
        color: #E65100;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
    }
    .metrics-table th, .metrics-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .metrics-table th {
        background-color: #f2f2f2;
    }
    .metrics-table small {
        font-size: 0.8em;
        color: #666;
        display: block;
        line-height: 1.2;
        margin-top: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Self improving targeting based on Tru-values")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="big-bold-title">Select a Movie</p>', unsafe_allow_html=True)
        movie_story_key = st.selectbox("Choose a Movie", list(movie_stories.keys()), key="movie_select")
        st.markdown(f'<div class="content-box">{format_movie_story(movie_stories[movie_story_key]["story"])}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="big-bold-title">Select a Persona</p>', unsafe_allow_html=True)
        persona_key = st.selectbox("Choose a Persona", list(personas.keys()), key="persona_select")
        st.markdown(f'<div class="content-box">{format_persona(personas[persona_key])}</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<p class="big-bold-title">Past Viewed Content</p>', unsafe_allow_html=True)
        st.selectbox("Past viewed content", [persona_key], key="past_viewed_content_select", disabled=True)
        st.markdown(f'<div class="content-box">{format_past_viewed(past_viewed_contents[persona_key])}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_reco, col_eval, _ = st.columns([1, 1, 2])

    with col_reco:
        llm_reco = st.selectbox("Recommender Model", ["gpt-4-0613", "gpt-3.5-turbo", "gpt-4o-mini"], index=2, key="reco-model-select")

    with col_eval:
        llm_eval = st.selectbox("Evaluator Model", ["gpt-4-0613", "gpt-3.5-turbo", "gpt-4o-mini"], index=2, key="eval-model-select")

    col_generate, col_reset, _ = st.columns([1, 1, 2])

    with col_generate:
        generate_button = st.button("Generate Recommendation")

    with col_reset:
        reset_button = st.button("Reset", key="reset_button", help="Clear all results and start over")

    if reset_button:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if generate_button:
        if "open_api_key" in st.secrets:
            api_key = st.secrets["open_api_key"]
            with st.spinner("Generating recommendation..."):
                openai_recommender = OpenAIRecommender(movie_stories[movie_story_key]["story"],
                                                    personas[persona_key],
                                                    past_viewed_contents[persona_key],
                                                    recommender_model=llm_reco,
                                                    evaluator_model=llm_eval,
                                                    n_past_recommendations=3)
                recommender = MovieRecommender(openai_recommender)

                update_placeholder = st.empty()

                for iteration, one_liner, tailored_message, metrics in recommender.run_recommendation_loop():
                    with update_placeholder.container():
                        st.markdown(f"### Iteration {iteration}")
                        st.markdown(f"**One-liner:** {one_liner}")
                        st.markdown(f"**Tailored Message:** {tailored_message}")
                        st.progress(iteration / recommender.max_iterations)

                        st.write("Current Metrics:")

                        total_score = metrics.get('total_score', 'N/A')
                        st.markdown(f"**Total Score:** <span style='color: red; font-weight: bold;'>{total_score:.2f}</span>", unsafe_allow_html=True)
                        
                        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)

                        # Add explanations for each metric
                        metric_explanations = {
                            'iteration': 'Current iteration number in the recommendation process.',
                            'total_score': 'Overall score combining all individual metric scores.',
                            'moving_avg': 'Average of recent scores to track improvement trends.',
                            'one_liner_relevance': 'How well the one-liner aligns with the persona and movie.',
                            'one_liner_emotionalImpact': 'Emotional resonance of the one-liner with the target audience.',
                            'one_liner_clarity': 'Clarity and understandability of the one-liner message.',
                            'one_liner_creativity': 'Originality and inventiveness of the one-liner.',
                            'tailored_message_relevance': 'Alignment of the tailored message with persona and movie.',
                            'tailored_message_callToAction': 'Effectiveness in motivating the audience to take action.',
                            'tailored_message_persuasiveness': 'Power of the message to convince the target audience.',
                            'tailored_message_specificity': 'Level of detail and precision in the tailored message.',
                            'diversity_score': 'Variety and uniqueness compared to previous recommendations.',
                            'persona_alignment_score': 'How well the recommendation matches the persona\'s interests.',
                            'cultural_relevance_score': 'Relevance to current cultural trends and values.',
                            'overall_improvement': 'Degree of enhancement compared to previous iterations.'
                        }

                        # Create HTML for the table with explanations
                        html_table = "<table class='metrics-table'>"
                        html_table += "<tr><th>Metric</th><th>Value</th><th>Explanation</th></tr>"
                        for _, row in metrics_df.iterrows():
                            metric = row['Metric']
                            value = row['Value']
                            explanation = metric_explanations.get(metric, "")
                            html_table += f"<tr><td>{metric}</td><td>{value}</td><td><small>{explanation}</small></td></tr>"
                        html_table += "</table>"

                        st.markdown(html_table, unsafe_allow_html=True)

            st.success("Recommendation generation complete!")

            st.session_state.recommender = recommender
            st.session_state.best_iteration = max(recommender.metrics_history, key=lambda x: x['total_score'])

            if 'recommender' in st.session_state and 'best_iteration' in st.session_state:
                recommender = st.session_state.recommender
                best_iteration = st.session_state.best_iteration

                st.markdown('<p class="big-bold-title">Best Performing Recommendations</p>', unsafe_allow_html=True)
                st.write(f"Iteration: {best_iteration['iteration']}")
                st.write(f"Total Score: {best_iteration['total_score']:.2f}")
                st.write(f"Best One-liner: {recommender.best_one_liner}")
                st.write(f"Best Tailored Message: {recommender.best_tailored_message}")

                col1, col2 = st.columns(2)
                with col1:
                    recommender.plot_metrics(["one_liner_relevance", "one_liner_emotionalImpact"], "One-liner Metrics")
                    recommender.plot_metrics(["persona_alignment_score", "cultural_relevance_score"], "Alignment Scores")
                with col2:
                    recommender.plot_metrics(["tailored_message_persuasiveness", "tailored_message_callToAction"], "Tailored Message Metrics")
                    recommender.plot_metrics(["total_score", "moving_avg"], "Total Score and Moving Average")
        else:
            st.error("OpenAI API key not found in secrets. Please add it to continue.")

if __name__ == "__main__":
    main()