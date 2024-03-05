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
Then, in the `Home.py` file, add this below the right section:

```python
if st.button("Your Page Title"):
    switch_page("Your Page Title")
```

Finally, you can run `python3 build.py` to add the "Show code" button to your page.
