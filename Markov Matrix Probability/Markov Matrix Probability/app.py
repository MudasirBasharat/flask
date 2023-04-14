from fractions import Fraction
from typing import List
from flask import Flask, render_template, request

app = Flask(__name__)

def solution(m: List[List[int]]) -> List[Fraction]:
    n = len(m)
    q = [[Fraction(m[i][j], sum(m[i])) if sum(m[i]) != 0 else Fraction(0) for j in range(n)] for i in range(n)]
    r = [q[i][i] for i in range(n)]
    f = [Fraction(0) for i in range(n)]
    f[0] = 1
    for i in range(100):
        f = [sum(q[j][i]*f[i] for i in range(n)) for j in range(n)]
    return f

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            matrix = [[int(x) for x in row.split(',')] for row in request.form['matrix'].split('\n')]
            result = [frac.numerator for frac in solution(matrix)]
            print(result)  # check the value of the result variable
            return render_template('index.html', result=result)
        except (ValueError, IndexError):
            # Invalid input format, display error message
            error_msg = "Invalid input format. Please enter a matrix of integers separated by commas, with one row per line."
            return render_template('index.html', error_msg=error_msg)
    else:
        return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)
