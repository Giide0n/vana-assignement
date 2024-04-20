# VANA Assignment - Gradient Descent

**Dozent:** Dr. Stefan Hackstein

Das folgende Video dient als [Leitfaden](https://tube.switch.ch/videos/5EVhmf2DcA) zur Aufgabe.

## Assignment: Gradient Descent

Das Ziel dieser Aufgabe besteht darin, dass Sie ein grundlegendes Verständnis für numerische Näherungsverfahren in
höheren Dimensionen erlangen, insbesondere für den Gradient Descent und dessen praktische Anwendung. Hierfür sollen Sie
ein Jupyter Notebook erstellen und das MNIST Dataset laden und erkunden. Anschließend sollen Sie ein neuronales Netzwerk
erstellen und trainieren, um die Bilder korrekt zu klassifizieren. Es dürfen nur die angegebenen Python Pakete verwendet
werden.

### Erlaubte Pakete

- numpy
- matplotlib
- Built-in Pakete (time, sys, math, ...)
- torchvision (nur für Aufgabe 1)

### Nicht erlaubte Pakete

- Alle anderen 3rd-Party Pakete, insbesondere PyTorch, Tensorflow, Keras, scikit-learn, torchvision (außerhalb von
  Aufgabe 1)

### Aufgaben

#### Aufgabe 1

Laden Sie das MNIST-Dataset (Training und Test) mithilfe des torchvision-Pakets (Verwenden Sie das torchvision Paket nur
für diese Aufgabe) und verwenden Sie matplotlib, um sich einen Überblick über die Daten zu verschaffen. Beschreiben Sie
die grundlegenden Eigenschaften des Datensets, z.B. wie viele und welche Daten es enthält und wie diese verteilt sind.

#### Aufgabe 2

Erstellen Sie eine Klasse für ein lineares Layer mit beliebig vielen Knoten. Implementieren Sie darin getrennte Methoden
für Forward-Pass, Backward-Pass und Parameter-Update mithilfe von numpy. Schreiben Sie geeignete Unittests, um die
Funktionsweise dieser Funktionen zu prüfen. Schreiben Sie insbesondere einen expliziten Test, für ein Layer mit 2
Knoten, welches als Input 2 Datensätzen zu je zwei 2 floats erhält. Wählen Sie dazu unterschiedliche feste Werte für
Input, initiale Gewichte und Lernrate. Dann berechnen sie von Hand die Ergebnisse von Forward, Backward und Update und
testen damit ihre Implementation. Legen Sie die Berechnung der Ergebnisse ihrer Lösung bei.

#### Aufgabe 3

Erstellen Sie ein neuronales Netzwerk mit 3 Hidden Layern mit gleicher Anzahl Knoten und einem Output Layer mit 10
Knoten. Verwenden Sie dazu die in Aufgabe 2 implementierte Klasse. Das Ziel ist die korrekte Klassifizierung aller
Ziffern. Bereiten Sie die Trainingsloop und alles dafür benötigte vor, um das Netzwerk darauf zu trainieren. Das heißt,
jeder Output Knoten wird einer Ziffer zugeordnet; der Output soll 1 für diese Ziffer und 0 für alle anderen Ziffern
sein. Das Netzwerk soll auf den Trainingsdaten trainieren und auf den Testdaten evaluiert werden. Achten Sie beim
Training darauf, nicht auf dem gesamten Datensatz gleichzeitig zu trainieren, sondern stückeln Sie diesen in kleine
Portionen (batches), auf denen nacheinander trainiert wird. Erläutern Sie kurz, warum das nötig ist. Verwenden Sie eine
geeignete Kosten-Funktion sowie Evaluations-Funktion und geben Sie deren mathematische Definition an. Begründen Sie Ihre
Wahl dieser Funktion und diskutieren Sie kurz eine weitere Option für Kosten und Evaluation mit einer Abwägung der Vor-
und Nachteile.

#### Aufgabe 4

Trainieren Sie das Netzwerk mit verschiedenen Lernraten (0.01 - 1) und Größen der Hidden Layer (4, 8, 16). Verfolgen Sie
während des Trainings die Entwicklung der Kosten- und Evaluations-Funktionen sowohl auf Trainings- als auch auf
Testdaten. Interpretieren Sie die Ergebnisse des Netzwerks und entscheiden Sie, welche Wahl von Lernrate und Hidden
Layer-Größe die Beste ist. Begründen Sie Ihre Wahl.

### Künstliche Intelligenz

Für diese Aufgaben dürfen KI Tools wie ChatGPT oder Github Copilot als Ressource genutzt werden, um Fragen zu stellen
oder bei Problemen Unterstützung zu erhalten. Voraussetzung ist, dass Sie transparent kommunizieren, wo und wie Sie
diese Tools eingesetzt haben und welche Verbesserungen nötig waren. Legen Sie Ihrer Abgabe einen Screenshot eines
relevanten Chats bei, der Ihren Umgang mit KI zeigt, und verfassen Sie eine kurze Reflexion über Ihren Umgang mit KI: Wo
hat es geholfen? Wo war es hinderlich?

### Eigenverantwortung

Diese Aufgabe zählt als unbegleitetes Selbststudium. Es wird also von Ihnen erwartet, sich die nötigen Informationen und
Verständnis eigenständig anzueignen, sowie diesen Prozess zu planen. Ich empfehle Ihnen, sich jede Woche zumindest 1-2
Stunden mit dieser Aufgabe zu beschäftigen und die Bearbeitung nicht aufzuschieben. Als Dozent stehe ich Ihnen auf
Anfrage (im Unterricht und per Mail) gerne als Ressource zur Verfügung, z.B. mit individuellem Feedback, Erläuterungen
und Diskussion. Kommen Sie gerne einfach auf mich zu.

### Hilfreiche Ressourcen

- Mehr zu Gradient Descent und Neuronalen Netzwerken
- Neuronale Netzwerke, erklärt
  von [3blue1brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- Jupyter Notebook Cloud Service: [Google Colab](https://colab.research.google.com/)
- Tutorials: Build Neural Network in Numpy:
    - [Medium.com](https://medium.com/@waleedmousa975/building-a-neural-network-from-scratch-using-numpy-and-math-libraries-a-step-by-step-tutorial-in-608090c20466)
    - [TowardsDataScience.com](https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795)
    - [TowardsDataScience.com](https://towardsdatascience.com/coding-a-neural-network-from-scratch-in-numpy-31f04e4d605)
    - [Kaggle.com](https://www.kaggle.com/code/atrisaxena/neural-network-from-scratch-using-numpy)

## Auswertungskriterien

Diese Auswertungskriterien bilden die Grundlage der Bewertung Ihrer Abgabe. Beachten Sie insbesondere den Malus, welcher
bei Abgaben droht, die nicht dem Mindestanspruch genügen. Die Bewertung findet anonym durch Ihre Kommilitonen statt,
welche Ihre Abgabe sichten und dazu diese Kriterien mit Ja oder Nein beantworten. Die Note einer Bewertung ergibt sich
linear aus dem Anteil der mit Ja beantworteten Kriterien. Jede Abgabe wird von 3 anderen Gruppen bewertet. Die
letztendliche Note ergibt sich als das Mittel dieser 3 Bewertungen. Wenn 2 von 3 Bewertungen, die von Ihrer Gruppe
abgegeben wurden, mit den letztendlich vergebenen Noten übereinstimmen (Fehlertoleranz: ≤ 0.2), erhält ihre Gruppe einen
Bonus von 0.2 auf die Note des Assignments.

### Dataset

1. Die Trainings- und Testdaten des MNIST-Datasets sind korrekt und nachvollziehbar mithilfe des torchvision-Pakets
   geladen worden.
2. Die Visualisierungen der Daten sind gut verständlich und repräsentativ.
3. Die grundlegenden Eigenschaften (Format, Typ, Verteilung) des MNIST-Datasets werden richtig beschrieben.

### Linear Layer

4. Die Klasse für ein lineares Layer wurde mit beliebig vielen Knoten korrekt und nachvollziehbar implementiert.
5. Es wurden geeignete Unittests geschrieben, um die Funktionsweise der Klasse zu prüfen, inklusive des expliziten
   Tests.
6. Die Rechnung zur expliziten Überprüfung von forward, backward und update ist übersichtlich, nachvollziehbar und
   korrekt.

### Klassifikationsmodell

7. Das neuronale Netzwerk wurde mithilfe der zuvor erstellten Klasse mit 3 Hidden Layern gleicher, frei wählbarer Größe
   und 10 Outputs korrekt und nachvollziehbar implementiert.
8. Das Training wurde in einer übersichtlichen Funktion korrekt vorbereitet, welche mit verschiedenen Parametern für das
   Netzwerk aufgerufen werden kann.
9. Die Erklärung, warum in batches trainiert werden muss, ist richtig, kurz und schlüssig.
10. Geeignete Kosten- und Evaluations-Funktionen wurden verwendet.
11. Die Wahl wurde begründet und mit anderen möglichen Funktionen verglichen.
12. Die mathematische Definition der verwendeten Kosten-Funktion und Evaluations-Funktion ist korrekt angegeben (
    gerendert in LaTeX).
13. Die geeignete Kosten-Funktion und Evaluations-Funktion wurde korrekt und nachvollziehbar implementiert.

### Training

14. Das Netzwerk wird korrekt und nachvollziehbar auf den Trainingsdaten trainiert (Trainingskosten sind monoton
    fallend).
15. Es wurden verschiedene Kombinationen von Lernraten und Größen des Hidden Layers sinnvoll ausprobiert.
16. Die Entwicklung der Kosten- und Evaluations-Funktionen wurden auf Trainings- und Testdatensätzen korrekt verfolgt
    und so dargestellt, dass verschiedene Modelle leicht vergleichbar sind.
17. Die Wahl von Lernrate und Hidden Layer-Größe wurde nachvollziehbar entschieden und begründet.

### Form

18. Das Notebook ist übersichtlich strukturiert und bietet eine Leseführung.
19. Die Ergebnisse werden gut verständlich kommuniziert und kritisch evaluiert.
20. Die Grafiken sind vollständig beschriftet und ohne weitere Erläuterung verständlich.
21. Der Code ist gut strukturiert sowie verständlich und angemessen kommentiert.
22. Die Ergebnisse werden am Ende des Notebooks so zusammengefasst, dass diese Zusammenfassung eigenständig verständlich
    ist.
23. Das Lerntagebuch ist kurz und verständlich geschrieben, zeigt den Lernfortschritt auf und macht zusammen mit den
    Kommentaren deutlich, wie und wofür ChatGPT und andere KI-Tools verwendet wurden.
24. Screenshot und Reflexion über KI-Nutzung sind aufschlussreich.

### Malus

- **-5** Es wurden nicht erlaubte Pakete verwendet.
- **-5** Das Notebook lässt sich nicht komplett und fehlerfrei in weniger als 5 Minuten ausführen.
- **-5** Der Text weist eine hohe Anzahl an grammatikalischen und Rechtschreibfehlern auf, was die Verständlichkeit
  erheblich beeinträchtigt.
- **-5** Die Schlussfolgerungen sind nicht durch die präsentierten Argumente und Daten gestützt oder erscheinen
  willkürlich.
- **-5** Der Inhalt des Textes ist teilweise nicht kohärent und wirkt wie automatisch generiertes Füllmaterial ohne
  klaren Bezug zum Thema.
