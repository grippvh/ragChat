import os

from langchain_ollama import ChatOllama

from config import Config
from llama_index_agent import RagChat

EVAL_PROMPT = """
Question: {question}
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer strictly with 'true' or 'false') Does the actual response match the expected response? 
"""

#TODO: rewrite according to new logic

class RagChatTest:
    def __init__(self):
        self.rag_chat = RagChat(Config.TEST_DB_PATH)
        self.test_data_folder = "test_data"

    def setup_test_environment(self):
        # Scan the test_data folder and ingest all PDF files
        if not os.path.exists(self.test_data_folder):
            raise FileNotFoundError(f"The folder '{self.test_data_folder}' does not exist.")

        pdf_files = [file for file in os.listdir(self.test_data_folder) if file.endswith(".pdf")]

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in the '{self.test_data_folder}' folder.")

        for file_name in pdf_files:
            file_path = os.path.join(self.test_data_folder, file_name)
            self.rag_chat.ingestor.ingest_file(file_path)
            print(f"Test PDF '{file_path}' ingested successfully.")

    def query_and_validate(self, question: str, expected_response: str, expect_success: bool = True):
        response_text = self.rag_chat.ask(question)

        print("\nQuestion:", question)
        print("Expected Response:", expected_response)

        prompt = EVAL_PROMPT.format(
            question=question, expected_response=expected_response, actual_response=response_text
        )

        evaluation_result = ChatOllama(model=Config.MODEL_NAME).invoke(prompt)
        evaluation_result_cleaned = evaluation_result.content.strip().lower()

        is_success = "true" in evaluation_result_cleaned

        if is_success:
            print("\033[92m" + f"Response: {response_text}" + "\033[0m")
        else:
            print("\033[91m" + f"Response: {response_text}" + "\033[0m")

        # Check if the result matches the expectation
        if is_success:
            return True, response_text
        else:
            return False, response_text

    def test_us_constitution_preamble(self):
        return self.query_and_validate(
            question="What is the purpose of the United States Constitution according to the Preamble?",
            expected_response="To form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity."
        )

    def test_us_constitution_article_ii_section_1(self):
        return self.query_and_validate(
            question="Who holds the executive power according to Article II, Section 1 of the U.S. Constitution?",
            expected_response="The executive Power shall be vested in a President of the United States of America."
        )

    def test_us_constitution_amendment_xiii(self):
        return self.query_and_validate(
            question="What does Amendment XIII of the Constitution abolish?",
            expected_response="Amendment XIII abolishes slavery and involuntary servitude, except as punishment for a crime."
        )

    def test_us_constitution_article_iii_section_1(self):
        return self.query_and_validate(
            question="Where is the judicial power of the United States vested according to Article III, Section 1?",
            expected_response="The judicial power of the United States is vested in one supreme Court and in such inferior Courts as the Congress may ordain and establish."
        )

    def test_us_constitution_amendment_xix(self):
        return self.query_and_validate(
            question="What right does Amendment XIX of the Constitution guarantee?",
            expected_response="Amendment XIX guarantees the right to vote shall not be denied or abridged on account of sex."
        )

    # Incorrect Test Cases

    def test_incorrect_us_constitution_preamble(self):
        return self.query_and_validate(
            question="What is the purpose of the United States Constitution according to the Preamble?",
            expected_response="To create a weak federal government and promote states' rights."
        )

    def test_incorrect_us_constitution_article_i_section_1(self):
        return self.query_and_validate(
            question="What powers are granted by Article I, Section 1 of the U.S. Constitution?",
            expected_response="No power is granted to the President of the United States."
        )

    # ticket to ride tests

    def test_game_end(self):
        return self.query_and_validate(
            question="When do you end game of Ticket to Ride?",
            expected_response="When one player’s stock of colored plastic trains gets down to only 0,1 or 2 trains left "
                              "at the end of his turn, each player, including that player, gets one final turn. "
                              "The game then ends and players calculate their final scores.",
        )

    def test_turn_sequence(self):
        return self.query_and_validate(
            question="What actions can a player take during their turn in Ticket to Ride?",
            expected_response="On a player's turn, they can do one of the following: draw Train cards, draw routes from routes deck, or draw Destination Tickets.",
        )

    def run_tests(self):
        # Set up the test environment
        self.setup_test_environment()

        # Store test results
        success_tests = []
        failed_tests = []

        # Running the tests and collecting results
        tests = [
            ("US Constitution Preamble", self.test_us_constitution_preamble, True),
            ("US Constitution Article II Section 1", self.test_us_constitution_article_ii_section_1, True),
            ("US Constitution Amendment XIII", self.test_us_constitution_amendment_xiii, True),
            ("US Constitution Article III Section 1", self.test_us_constitution_article_iii_section_1, True),
            ("US Constitution Amendment XIX", self.test_us_constitution_amendment_xix, True),
            ("Incorrect US Constitution Preamble", self.test_incorrect_us_constitution_preamble, False),
            ("Incorrect US Constitution Article I Section 1", self.test_incorrect_us_constitution_article_i_section_1, False),
        ]

        for test_name, test_func, expect_success in tests:
            result, response = test_func()
            if result == expect_success:
                success_tests.append(test_name)
            else:
                failed_tests.append((test_name, response))

        # Display results
        print("\nTest Summary:")
        print("-------------")

        if success_tests:
            print(f"Successful Tests ({len(success_tests)}):")
            for test_name in success_tests:
                print(f" - {test_name}")

        if failed_tests:
            print(f"\nFailed Tests ({len(failed_tests)}):")
            for test_name, response in failed_tests:
                print(f" - {test_name}")
                print(f"   Response: {response}")

        if not failed_tests:
            print("\nAll tests completed successfully.")
        else:
            print("\nSome tests failed. Please review the responses above.")


# Run the tests
if __name__ == "__main__":
    test_runner = RagChatTest()
    test_runner.run_tests()
