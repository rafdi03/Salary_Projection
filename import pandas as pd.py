import pandas as pd
import locale
import matplotlib.pyplot as plt

locale.setlocale(locale.LC_ALL, 'id_ID')

df = pd.read_csv("Salary_Data.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, 1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

plt.scatter(X_train, y_train, color="green")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Tahun Pengalaman VS Gaji")
plt.xlabel("Tahun Pengalaman")
plt.ylabel("Gaji") 
plt.show()

years_of_experience = float(input("Masukkan jumlah tahun pengalaman: "))
predicted_salary = regressor.predict([[years_of_experience]])
formatted_salary = locale.currency(predicted_salary[0], grouping=True)
print(f"Untuk seseorang dengan {years_of_experience} tahun pengalaman, gaji yang diprediksi adalah: {formatted_salary}")
