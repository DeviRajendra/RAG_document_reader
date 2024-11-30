
from retriever import Retriever


class PromptGenerator():
    def __init__(self):
        self.pdf_path =  r"C:\work_drv_windows\ML_BOOKS\2011_Book_ComputerVision.pdf"
        self.get_book_path()
        print(self.pdf_path)
        self.retriever = Retriever(self.pdf_path)

    def get_response(self,prompt):

        response = self.retriever.retrieve_content(prompt)
        return response

    def get_book_path(self):
        # Default book path
        default_path = self.pdf_path
        print(f"Default book path: {default_path}")
        
        choice = input("Do you want to use a different path? (yes/no): ").strip().lower()
        
        if choice in ['yes', 'y']:

            new_path = input("Please enter the full path to your book: ").strip().replace('"','').replace("'",'')

            import os
            if os.path.exists(new_path):
                print(f"Book path set to: {new_path}")
                self.pdf_path = new_path
            else:
                print("Invalid path! Using default path instead.")
        elif choice in ['no', 'n']:
            print(f"Using default path: {default_path}")
        else:
            print("Invalid input! Using default path.")

if __name__ == "__main__":
    pg = PromptGenerator()
    while True:
        prompt = input("Enter your prompt(or type 'exit' to quit) \n")
        if prompt.lower() == "exit":
            print('Goodbye')
            break
        response = pg.get_response(prompt)
        print(response['answer'])