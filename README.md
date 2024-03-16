# Mathematics Explorer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mathviz.streamlit.app)

This app allows you to explore various mathematical/statistical concepts and
visualizations, along with machine learning models and algorithms.

## Install

```bash
git clone https://github.com/killian31/MathViz.git; cd MathViz
pip install -r requirements.txt
```

## Usage

Run `streamlit run Home.py`

## Contribution

We welcome contributions from the community!

To add your app, add your file app in the folder `pages/`, named like `pages/Your_Page_Title.py`. Right after your imports, add the following line in your script:

```python
st.set_page_config(page_title="Your Page Title")
```

Before continuing, please make sure you have added the python libraries your app uses to the `requirements.txt` file, otherwise your app will not be able to run on the cloud.

Then, in the `Home.py` file, add this below the right section:

```python
<section>_menu.page_link(
    "pages/Your_Page_Title.py",
    label="Your Page Title",
    use_container_width=True
)
```

Paste the following piece of code to apply the styling for buttons:

```python
st.markdown(
    """ <style>
                button {
                    background-color: #f63366;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    transition-duration: 0.2s;
                    cursor: pointer;
                    border-radius: 12px;
                }
                button:hover {
                    background-color: white;
                    color: black;
                    border: 2px solid #f63366;
                }
            </style>""",
    unsafe_allow_html=True,
)
```

Finally, you can run `python3 build.py` to add the "Show code" button to your page.
