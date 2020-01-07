# app.py

# from scripts.__init__ import *  # import pandas, numpy, scipy, json, flask
from scripts.functions import *  # import all functions and packages

app = Flask(__name__)

vector_beer_rating = []


@app.route('/')
def hello():  # call method hello
    return render_template('home.html')


@app.route("/cocktail")  # at the end point /
def page():  # call method hello
    return render_template('cocktails_select.html')


@app.route("/beer", methods=["GET", "POST"])
def selecting_beers():
    global vector_beer_rating

    if request.method == "POST":
        # New added beer
        beer = request.form['beer_choice']
        rating = request.form['beer_rating']

        # Update the list and count
        vector_beer_rating.append((str(beer), int(rating)))
        # vector_beer_rating = list(set(vector_beer_rating))  # Useless if available list is already filtered ?
        out = len(vector_beer_rating)

        # Difference between initial list and selected list
        beers_available_ls = updated_list(beer_ls, [x[0] for x in vector_beer_rating])

    elif request.method == "GET":
        # Get empty values for the template
        vector_beer_rating = []
        out = 0

        # Difference between initial list and selected list
        beers_available_ls = beer_ls.copy()

    return render_template('beers_select.html', data=beers_available_ls, selection=vector_beer_rating, reco=out)


@app.route("/beer_recommendation", methods=["GET", "POST"])
def recommending_beers():
    #Get used data
    global vector_beer_rating

    top20reco, id, selected = launch_reco(selection_beer=vector_beer_rating, df_beers=df_beers, df_matrix=df_pivot)

    if request.method == "GET":
        predictions = top20reco[['beer_name', 'brewery_name', 'beer_abv', 'beer_style']].head(5)

        return render_template('beers_reco.html', selection=selected, value_user_id=id,
                               top_beer=predictions, displayed_rows=5,
                               tables=[predictions.to_html(classes='data', index=False)])

    if request.method == "POST":
        # Nb rows to return
        rows = request.form['top_return']
        predictions = top20reco.head(int(rows))

        return render_template('beers_reco.html', selection=selected, value_user_id=id,
                               top_beer=predictions, displayed_rows=int(rows),
                               tables=[predictions.to_html(classes='data', index=False)])


if __name__ == "__main__":
    app.run(debug=True)

#
