# AI-Machine-Learning-Assignment-Week-2-Assignment-AI-for-Sustainable-Development

# ai_ml_quiz_answers.py

"""
This script provides the correct answers to a series of multiple-choice questions
related to Artificial Intelligence and Machine Learning concepts.

It serves as a consolidated answer key for review.
"""

def print_question_and_answer(question_number, question, options, correct_option_letter, correct_option_text, explanation=None):
    """
    Helper function to print a question, its options, and the correct answer.
    """
    print(f"\n--- Question {question_number} ---")
    print(question)
    for letter, text in options:
        print(f"{letter}) {text}")
    print(f"\nCorrect Answer: {correct_option_letter}) {correct_option_text}")
    if explanation:
        print(f"Explanation: {explanation}")
    print("-" * 40)

# --- Questions and Answers ---

# Question 1: Vibe Coding Description
q1_question = "Which of the following best describes vibe coding?"
q1_options = [
    ("A", "A method of writing verbose code by strictly following syntax rules."),
    ("B", "A process where natural language prompts lead AI tools to generate or modify code."),
    ("C", "A technique that relies solely on manual coding without any AI assistance."),
    ("D", "An advanced debugging strategy to optimize existing code.")
]
q1_correct_letter = "B"
q1_correct_text = "A process where natural language prompts lead AI tools to generate or modify code."
q1_explanation = "Vibe coding uses natural language to guide AI in generating or modifying code, focusing on intent rather than strict syntax."
print_question_and_answer(1, q1_question, q1_options, q1_correct_letter, q1_correct_text, q1_explanation)


# Question 2: Primary Benefit of Vibe Coding
q2_question = "What is one primary benefit of vibe coding?"
q2_options = [
    ("A", "It eliminates the need for any pre-planning or testing."),
    ("B", "It speeds up the development process by transforming ideas into code rapidly."),
    ("C", "It increases the complexity of code by adding unnecessary boilerplate."),
    ("D", "It restricts creativity by enforcing rigid syntax rules.")
]
q2_correct_letter = "B"
q2_correct_text = "It speeds up the development process by transforming ideas into code rapidly."
q2_explanation = "Vibe coding's main benefit is rapidly converting ideas expressed in natural language into functional code, accelerating development."
print_question_and_answer(2, q2_question, q2_options, q2_correct_letter, q2_correct_text, q2_explanation)


# Question 3: Multimedia Documentation Tools
q3_question = "In a composite workflow for producing a full-stack application, which tool combination is best for multimedia documentation of the project?"
q3_options = [
    ("A", "Cursor AI and Bolt.new"),
    ("B", "Lovable AI and Cursor AI"),
    ("C", "Pictory and Synthesia"),
    ("D", "Bolt.new and Lovable AI")
]
q3_correct_letter = "C"
q3_correct_text = "Pictory and Synthesia"
q3_explanation = "Pictory and Synthesia are AI tools specifically designed for video and multimedia generation from text or existing content, ideal for rich documentation."
print_question_and_answer(3, q3_question, q3_options, q3_correct_letter, q3_correct_text, q3_explanation)


# Question 4: Plan-Verify-Execute Benefit
q4_question = "How does using a 'plan-verify-execute' prompt structure benefit vibe coding?"
q4_options = [
    ("A", "It allows the AI to execute code without any human oversight."),
    ("B", "It enforces a rigid development process that discourages changes."),
    ("C", "It creates checkpoints by having the AI outline its planned modifications, reducing the risk of unintended changes."),
    ("D", "It eliminates the need for any testing or iterative feedback.")
]
q4_correct_letter = "C"
q4_correct_text = "It creates checkpoints by having the AI outline its planned modifications, reducing the risk of unintended changes."
q4_explanation = "The 'plan-verify-execute' structure provides human oversight before execution, reducing errors and unintended changes."
print_question_and_answer(4, q4_question, q4_options, q4_correct_letter, q4_correct_text, q4_explanation)


# Question 5: Guardrails Benefit
q5_question = "What does setting up guardrails in vibe coding prompt engineering help ensure?"
q5_options = [
    ("A", "That AI only performs modifications explicitly specified in the prompt, minimizing unintended changes."),
    ("B", "That AI is allowed to alter any part of the codebase without restrictions."),
    ("C", "That the entire project must be rewritten from scratch at each iteration."),
    ("D", "That manual code changes are completely eliminated in development.")
]
q5_correct_letter = "A"
q5_correct_text = "That AI only performs modifications explicitly specified in the prompt, minimizing unintended changes."
q5_explanation = "Guardrails define boundaries for AI actions, ensuring it sticks to specified modifications and prevents unintended alterations."
print_question_and_answer(5, q5_question, q5_options, q5_correct_letter, q5_correct_text, q5_explanation)


# Question 6: CI/CD Integration Benefit
q6_question = "How does integrating vibe-coded projects with cloud-based CI/CD pipelines benefit the development lifecycle?"
q6_options = [
    ("A", "It introduces unnecessary complexity without tangible improvements."),
    ("B", "It only benefits multimedia content creation and not traditional code."),
    ("C", "It replaces all the natural language processing aspects of vibe coding."),
    ("D", "It automates testing, containerization, vulnerability scanning, and deployment—ensuring rapid and secure production releases.")
]
q6_correct_letter = "D"
q6_correct_text = "It automates testing, containerization, vulnerability scanning, and deployment—ensuring rapid and secure production releases."
q6_explanation = "CI/CD pipelines automate critical steps like testing and deployment, speeding up the delivery of stable and secure applications generated via vibe coding."
print_question_and_answer(6, q6_question, q6_options, q6_correct_letter, q6_correct_text, q6_explanation)


# Question 7: Definition of AI
q7_question = "What best defines Artificial Intelligence (AI)?"
q7_options = [
    ("A", "Machines that can perform physical tasks like humans."),
    ("B", "Systems that simulate human intelligence to think and learn."),
    ("C", "Software designed exclusively for data storage."),
    ("D", "Tools used only for mathematical calculations.")
]
q7_correct_letter = "B"
q7_correct_text = "Systems that simulate human intelligence to think and learn."
q7_explanation = "AI aims to create machines capable of cognitive functions typically associated with human minds, such as reasoning, learning, and problem-solving."
print_question_and_answer(7, q7_question, q7_options, q7_correct_letter, q7_correct_text, q7_explanation)


# Question 8: Types of AI
q8_question = "Which of the following is a type of AI?"
q8_options = [
    ("A", "Cloud AI, Edge AI, Hybrid AI"),
    ("B", "Narrow AI, General AI, Superintelligent AI"),
    ("C", "Fast AI, Slow AI, Balanced AI"),
    ("D", "Simple AI, Complex AI, Adaptive AI")
]
q8_correct_letter = "B"
q8_correct_text = "Narrow AI, General AI, Superintelligent AI"
q8_explanation = "These terms categorize AI based on its capability and intelligence level: Narrow AI (current), General AI (human-level), and Superintelligent AI (beyond human)."
print_question_and_answer(8, q8_question, q8_options, q8_correct_letter, q8_correct_text, q8_explanation)


# Question 9: Siri/Alexa Example
q9_question = "A voice assistant like Siri or Alexa is an example of:"
q9_options = [
    ("A", "General AI"),
    ("B", "Superintelligent AI"),
    ("C", "Narrow AI"),
    ("D", "Self-aware AI")
]
q9_correct_letter = "C"
q9_correct_text = "Narrow AI"
q9_explanation = "Voice assistants are designed for specific tasks (e.g., answering questions, setting alarms) and do not possess human-level general intelligence or self-awareness."
print_question_and_answer(9, q9_question, q9_options, q9_correct_letter, q9_correct_text, q9_explanation)


# Question 10: AI in Software Engineering
q10_question = "Which AI application is most relevant to software engineering?"
q10_options = [
    ("A", "Autonomous driving systems"),
    ("B", "Automated code generation and testing"),
    ("C", "Medical diagnosis tools"),
    ("D", "Social media photo filters")
]
q10_correct_letter = "B"
q10_correct_text = "Automated code generation and testing"
q10_explanation = "AI applications like code generation (e.g., GitHub Copilot) and automated testing directly assist and enhance the software development process."
print_question_and_answer(10, q10_question, q10_options, q10_correct_letter, q10_correct_text, q10_explanation)


# Question 11: ML type using trial and error
q11_question = "Which machine learning type involves training models through trial and error?"
q11_options = [
    ("A", "Supervised Learning"),
    ("B", "Unsupervised Learning"),
    ("C", "Reinforcement Learning"),
    ("D", "Deep Learning")
]
q11_correct_letter = "C"
q11_correct_text = "Reinforcement Learning"
q11_explanation = "Reinforcement learning agents learn optimal behaviors by interacting with an environment and maximizing cumulative rewards through experimentation."
print_question_and_answer(11, q11_question, q11_options, q11_correct_letter, q11_correct_text, q11_explanation)


# Question 12: Critical Ethical Concern in AI
q12_question = "What ethical concern is critical when designing AI systems?"
q12_options = [
    ("A", "Ensuring bright color schemes in user interfaces."),
    ("B", "Minimizing hardware costs."),
    ("C", "Avoiding bias and ensuring fairness."),
    ("D", "Using the fastest programming language.")
]
q12_correct_letter = "C"
q12_correct_text = "Avoiding bias and ensuring fairness."
q12_explanation = "Bias in training data can lead AI systems to make discriminatory decisions, making fairness a paramount ethical concern in AI design."
print_question_and_answer(12, q12_question, q12_options, q12_correct_letter, q12_correct_text, q12_explanation)


# Question 13: ML type with labeled data
q13_question = "Which machine learning paradigm uses labeled data to train models?"
q13_options = [
    ("A", "Reinforcement Learning"),
    ("B", "Unsupervised Learning"),
    ("C", "Supervised Learning"),
    ("D", "Semi-supervised Learning")
]
q13_correct_letter = "C"
q13_correct_text = "Supervised Learning"
q13_explanation = "Supervised learning trains models on datasets where each input is paired with a corresponding correct output label."
print_question_and_answer(13, q13_question, q13_options, q13_correct_letter, q13_correct_text, q13_explanation)


# Question 14: Primary Goal of Reinforcement Learning
q14_question = "What is the primary goal of reinforcement learning?"
q14_options = [
    ("A", "Group similar data points into clusters"),
    ("B", "Learn by interacting with an environment to maximize rewards"),
    ("C", "Predict outcomes using historical labeled data"),
    ("D", "Reduce data dimensionality")
]
q14_correct_letter = "B"
q14_correct_text = "Learn by interacting with an environment to maximize rewards"
q14_explanation = "Reinforcement learning aims for an agent to discover an optimal policy for making decisions that maximize its cumulative reward in an environment."
print_question_and_answer(14, q14_question, q14_options, q14_correct_letter, q14_correct_text, q14_explanation)


# Question 15: Non-linearity in Neural Networks
q15_question = "Which activation function is commonly used to introduce non-linearity in neural networks?"
q15_options = [
    ("A", "Linear"),
    ("B", "Sigmoid"),
    ("C", "Mean Squared Error"),
    ("D", "Gradient Descent")
]
q15_correct_letter = "B"
q15_correct_text = "Sigmoid"
q15_explanation = "Non-linear activation functions like Sigmoid, ReLU, or Tanh are essential for neural networks to learn complex, non-linear relationships in data."
print_question_and_answer(15, q15_question, q15_options, q15_correct_letter, q15_correct_text, q15_explanation)


# Question 16: Clustering Customer Data
q16_question = "Clustering customer data into groups based on purchasing behavior is an example of:"
q16_options = [
    ("A", "Supervised Learning"),
    ("B", "Reinforcement Learning"),
    ("C", "Unsupervised Learning"),
    ("D", "Deep Learning")
]
q16_correct_letter = "C"
q16_correct_text = "Unsupervised Learning"
q16_explanation = "Clustering involves finding hidden patterns or groupings in unlabeled data, which is characteristic of unsupervised learning."
print_question_and_answer(16, q16_question, q16_options, q16_correct_letter, q16_correct_text, q16_explanation)


# Question 17: Deep Learning Reliance
q17_question = "Deep Learning primarily relies on:"
q17_options = [
    ("A", "Decision trees and random forests"),
    ("B", "Layered artificial neural networks"),
    ("C", "Rule-based algorithms"),
    ("D", "Linear regression models")
]
q17_correct_letter = "B"
q17_correct_text = "Layered artificial neural networks"
q17_explanation = "Deep learning is defined by the use of deep (multi-layered) artificial neural networks to learn complex patterns and representations."
print_question_and_answer(17, q17_question, q17_options, q17_correct_letter, q17_correct_text, q17_explanation)


# Question 18: Core NLP Application
q18_question = "Which task is a core application of Natural Language Processing (NLP)?"
q18_options = [
    ("A", "Image recognition"),
    ("B", "Sentiment analysis of text"),
    ("C", "Predicting stock prices"),
    ("D", "Robot motion planning")
]
q18_correct_letter = "B"
q18_correct_text = "Sentiment analysis of text"
q18_explanation = "NLP focuses on enabling computers to understand, interpret, and generate human language, making sentiment analysis a direct application."
print_question_and_answer(18, q18_question, q18_options, q18_correct_letter, q18_correct_text, q18_explanation)


# Question 19: Overfitting Definition
q19_question = "Overfitting in machine learning occurs when:"
q19_options = [
    ("A", "The model performs well on training data but poorly on unseen data"),
    ("B", "The model is too simple to capture patterns"),
    ("C", "The dataset is too small"),
    ("D", "The learning rate is too low")
]
q19_correct_letter = "A"
q19_correct_text = "The model performs well on training data but poorly on unseen data"
q19_explanation = "Overfitting happens when a model memorizes the training data, including noise, leading to poor generalization on new data."
print_question_and_answer(19, q19_question, q19_options, q19_correct_letter, q19_correct_text, q19_explanation)


# Question 20: CNN Effectiveness
q20_question = "Convolutional Neural Networks (CNNs) are most effective for:"
q20_options = [
    ("A", "Time-series forecasting"),
    ("B", "Image recognition tasks"),
    ("C", "Text classification"),
    ("D", "Anomaly detection in tabular data")
]
q20_correct_letter = "B"
q20_correct_text = "Image recognition tasks"
q20_explanation = "CNNs are specifically designed with architectures (convolutional and pooling layers) that excel at processing and learning features from image data."
print_question_and_answer(20, q20_question, q20_options, q20_correct_letter, q20_correct_text, q20_explanation)


# Question 21: Metric NOT for Classification
q21_question = "Which metric is NOT typically used for classification model evaluation?"
q21_options = [
    ("A", "Accuracy"),
    ("B", "Precision"),
    ("C", "Mean Absolute Error (MAE)"),
    ("D", "Recall")
]
q21_correct_letter = "C"
q21_correct_text = "Mean Absolute Error (MAE)"
q21_explanation = "MAE is a regression metric used for continuous numerical predictions, whereas Accuracy, Precision, and Recall are used for categorical classification outcomes."
print_question_and_answer(21, q21_question, q21_options, q21_correct_letter, q21_correct_text, q21_explanation)


# Question 22: Framework for Dynamic Neural Networks
q22_question = "Which framework is known for its flexibility in building dynamic neural networks?"
q22_options = [
    ("A", "Scikit-learn"),
    ("B", "TensorFlow"),
    ("C", "PyTorch"),
    ("D", "Keras")
]
q22_correct_letter = "C"
q22_correct_text = "PyTorch"
q22_explanation = "PyTorch's 'define-by-run' (dynamic computational graph) approach provides high flexibility for research and building complex, dynamic neural network architectures."
print_question_and_answer(22, q22_question, q22_options, q22_correct_letter, q22_correct_text, q22_explanation)

print("\n--- End of Quiz Answers ---")
