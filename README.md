# ML-playground
 
## Example with Kaggle Titanic data

````
from ml_playground.data import DataContainer, TextTransformer
from ml_playground.modeller import Modeller
from sklearn.tree import DecisionTreeClassifier

data = DataContainer("../Kaggle/Titanic/train.csv")
data.apply("dropna")
data.transform("Sex", TextTransformer)
data.transform("Embarked", TextTransformer)
data.x = ["Sex", "Pclass", "Age", "SibSp", "Parch", "Embarked"]
data.y = "Survived"

model = Modeller(DecisionTreeClassifier)
model.fit(data)
```