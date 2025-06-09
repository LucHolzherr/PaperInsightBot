## Project Description
The PaperInsightBot takes as input the name of an academic paper and returns relevant information about each author of the paper. 
It gathers the information using the semantic scholar api and a web-search using the TavilyClient (https://www.tavily.com/).
A Langchain LLM agent is used to combine the search results in a concise summary. The output is the summary as html and as .txt file.
### Example
**Input:** Depth Pro: Sharp Monocular Metric Depth in Less Than a Second

**Output:**
> Vladlen Koltun is a renowned researcher in computer science, specializing in machine learning, robotics, computer vision, and image synthesis. He has over 71,643 citations and an h-index of over 109. His most cited work is on multi-scale context aggregation by dilated convolutions. Koltun has also contributed significantly to the development of CARLA, an open-source simulator for autonomous driving research. He is currently the director of an international research organization at Apple and was previously the Chief Scientist for Intelligent Systems at Intel. Koltun's work has been published in reputable journals and presented at prestigious conferences.
> 
> Alexey Bochkovskiy, also known as Aleksei Bochkovskii, is a prominent researcher in computer vision and object detection, with over 20,153 citations and an h-index of over 6. He is best known for his development of YOLOv4 and YOLOv7, which have set new standards for real-time object detection. Bochkovskiy is affiliated with Apple Inc. and Google Scholar's Academy. His work has been presented at prestigious conferences and he maintains an active presence on GitHub, where he shares his work on the application of the YOLOv4 object detection neural network.
> 
> Stephan R. Richter is a significant contributor to the field of computer vision, with over 3,496 citations and an h-index of over 11. His most cited work is "Playing for Data: Ground Truth from Computer Games". Richter is also the publisher and editor-in-chief of The Globalist, a daily online magazine. He leads a research team at Apple, where his interests include AR/VR, photorealistic rendering, sim2real, 3D scene understanding, and machine learning.
> 
> Amaël Delaunoy is a researcher specializing in computer vision and robotics, with over 466 citations and an h-index of over 11. His research contributions span various areas, including 3D modeling and reconstruction. Delaunoy's work has been widely recognized, with his research on minimizing the multi-view stereo reprojection error for triangular surface meshes and the development of a novel algorithm for automatic 3D reconstruction from images being particularly notable. He is currently affiliated with Apple and was previously associated with ETH Zurich.


## Install Instructions
Code was tested with Python 3.13. LangChain version 0.3.X requires Python >= 3.9.
1. Create a virtual environment, activate it and install required packages:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. In the .env file in the root directory, add your API keys for openai and tavily.
```
OPENAI_API_KEY=YOUR_KEY
TAVILY_API_KEY=YOUR_KEY
```
3. Run main.py with the config input argument:
```
python main.py --config config/default_config.yamls
```

## Todos
- [ ] Change from SemanticScholar API to google scholar API. Semantic Scholar is missing many papers and overall citation count is not accurate.
- [ ] Web search result verification: check if the person actually matches the author of the paper.
- [ ] Provide web search links from which the information was pulled.
