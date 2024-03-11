import os


def add_show_code_button():
    # get all files in ./pages/
    files = os.listdir("./pages/")
    # for each file, first get the raw code (ie the content of the file) and then add a button to show the code below the st.set_page_config
    for file in files:
        if file.endswith(".py"):
            with open(f"./pages/{file}", "r") as f:
                code = f.read()
                # get the index of the line where st.set_page_config ends, ie the line after the last line of the st.set_page_config block
                st_page_index = code.find("st.set_page_config")
                end_index = code.find(")", st_page_index)
                line_index = code.find("\n", end_index + 1)
                # add the button to show the code if it's not already there
                if "st.button('Show code')" not in code:
                    code = (
                        code[:line_index]
                        + f"\n# This is automatically generated, do not modify"
                        + f"\nif st.button('Show code'):\n    st.code('''{code}''')\n"
                        + code[line_index:]
                    )
                    # write the new code to the file
                    with open(f"./pages/{file}", "w") as f:
                        f.write(code)
                    print(f"Added show code button to {file}")


if __name__ == "__main__":
    add_show_code_button()
    print("Done!")
