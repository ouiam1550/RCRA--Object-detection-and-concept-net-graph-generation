from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import torch
import numpy as np
import requests
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
GRAPH_FOLDER = "static/graphs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# Charger le modèle YOLOv5
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

def get_conceptnet_relations(objects):
    G = nx.Graph()
    related_concepts = set()
    
    for obj in objects:
        url = f"https://api.conceptnet.io/query?node=/c/en/{obj}&rel=/r/RelatedTo&limit=5"
        response = requests.get(url).json()
        
        for edge in response.get("edges", []):
            start = edge["start"]["label"]
            end = edge["end"]["label"]
            if start in objects or end in objects:
                G.add_edge(start, end)
                related_concepts.update([start, end])
    
    return G, related_concepts

def estimate_scene(objects):
    scene_candidates = {
        "kitchen": {"table", "plate", "cup", "fork", "spoon", "knife"},
        "bedroom": {"bed", "pillow", "blanket", "lamp"},
        "office": {"laptop", "desk", "chair", "keyboard"},
        "street": {"car", "bicycle", "bus", "traffic light"},
        "parc": {"tree","plant"},
    }
    
    best_scene = "unknown"
    max_overlap = 0
    
    for scene, keywords in scene_candidates.items():
        overlap = len(objects & keywords)
        if overlap > max_overlap:
            max_overlap = overlap
            best_scene = scene
    
    return best_scene

def plot_graph(G, central_node, filename):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    
    G.add_node(central_node, size=3000)
    for node in list(G.nodes):
        if node != central_node:
            G.add_edge(central_node, node)
    
    sizes = [G.nodes[n].get("size", 1000) for n in G.nodes]
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=sizes, font_size=10)
    plt.savefig(filename)
    plt.close()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    if file:
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        result_path = os.path.join(RESULT_FOLDER, file.filename)
        graph_path = os.path.join(GRAPH_FOLDER, f"{file.filename}.png")

        file.save(image_path)

        # Détection d'objets
        img = Image.open(image_path)
        results = model(img)
        results.render()

        # Sauvegarde de l'image annotée
        detected_img = np.array(results.ims[0])
        cv2.imwrite(result_path, cv2.cvtColor(detected_img, cv2.COLOR_RGB2BGR))

        # Récupérer les objets détectés
        detected_objects = set(results.pandas().xyxy[0]['name'].dropna())
        
        # Obtenir le graphe de relations avec ConceptNet
        G, related_concepts = get_conceptnet_relations(detected_objects)
        scene = estimate_scene(detected_objects)
        
        # Ajouter la scène principale comme noeud central
        G.add_node(scene, size=4000)
        for obj in detected_objects.union(related_concepts):
            G.add_edge(scene, obj)
        
        plot_graph(G, scene, graph_path)
        
        return render_template("index.html", uploaded=True, filename=file.filename, objects=detected_objects, graph_file=f"{file.filename}.png", scene=scene)

@app.route("/static/results/<filename>")
def result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route("/static/graphs/<filename>")
def graph_image(filename):
    return send_from_directory(GRAPH_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
