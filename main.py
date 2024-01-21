#!/usr/bin/env python3.7

import sys, random
import numpy as np
from numpy.linalg import inv
from scipy import stats
import pandas as pd
import plotly.express as px, plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, State, Output, ctx
from dash.dash_table.Format import Format, Scheme, Group

app = Dash(__name__)
df = None
degree = None
model = None
initial_input = []

EMPTY = (None, '')

def main(*args):
	global app, df, degree, model, initial_input
	df = pd.read_csv('Inkindo.csv')
	degree = df['Degree'].unique()
	model = linear_regression(model_X(df), model_y(df))
	# model.info()
	initial_input.extend([
		{
			'Degree'       : 'Bachelor',
			'Experience'   : 3,
			'From Year'    : 2007,
			'To Year'      : 2023,
			'Mean Level'   : 0.95,
			'Predict Level': 0.95,
			'Color'        : 0
		}, {
			'Degree'       : 'Master',
			'Experience'   : 5,
			'From Year'    : 2007,
			'To Year'      : 2023,
			'Mean Level'   : 0.95,
			'Predict Level': 0.95,
			'Color'        : 1
		}, {
			'Degree'       : 'Doctorate',
			'Experience'   : 9,
			'From Year'    : 2007,
			'To Year'      : 2023,
			'Mean Level'   : 0.95,
			'Predict Level': 0.95,
			'Color'        : 2
		}
	])
	app_layout()
	app.run_server(debug=('debug' in args))
	print('Hello, world!')

def app_layout():
	global app, degree, initial_input
	app.layout = html.Div([html.Div([
		dash_table.DataTable(
			id='input_table',
			columns=[
				{'id': 'Degree'       , 'name': 'Degree'       , 'presentation': 'dropdown'},
				{'id': 'Experience'   , 'name': 'Experience'   , 'type': 'numeric'},
				{'id': 'From Year'    , 'name': 'From Year'    , 'type': 'numeric'},
				{'id': 'To Year'      , 'name': 'To Year'      , 'type': 'numeric'},
				{'id': 'Mean Level'   , 'name': 'Mean Level'   , 'type': 'numeric'},
				{'id': 'Predict Level', 'name': 'Predict Level', 'type': 'numeric'},
				{'id': 'Color'  , 'name': 'Color'  , 'type': 'numeric'}
			],
			data=initial_input,
			editable=True,
			row_deletable=True,
			dropdown={'Degree': {'options': [{'label': d, 'value': d} for d in degree]}}
		),
		html.Br(),
		html.Hr(), html.Br(),
		html.Button('Add Input', id='add_input', n_clicks=0),
		html.Br(), html.Br(),
		dcc.Checklist(
			[
				{'label': 'Draw trace on data', 'value': 'trace'},
				{'label': 'Draw scatter on analysis', 'value': 'scatter'},
				{'label': 'Draw mean confidence interval', 'value': 'mean'},
				{'label': 'Draw prediction interval', 'value': 'pred'},
			],
			['mean', 'pred'],
			id='scatter_toggle',
			labelStyle={'display': 'block'}
		),
		html.Br(), html.Hr(),
		html.Div(id='output')
	], style={'width': '100vmin'})], style={'display': 'flex', 'justify-content': 'center'})

@app.callback(
	Output('input_table', 'data'),
	Input('add_input', 'n_clicks'),
	Input('input_table', 'data'),
	State('input_table', 'columns')
)
def add_input(n_clicks, rows, columns):
	global EMPTY
	if ctx.triggered_id == 'add_input' and n_clicks > 0:
		rows.append({c['id']: '' for c in columns})
	for row in rows:
		if row['Mean Level'] in EMPTY: row['Mean Level'] = 0.95
		if row['Predict Level'] in EMPTY: row['Predict Level'] = 0.95
		if row['Color'] in [None, '']:
			used_color = set()
			for r in rows:
				used_color.add(r['Color'])
			i = 0
			while True:
				if i not in used_color:
					row['Color'] = i
					break
				i += 1
	return rows

@app.callback(
	Output('output', 'children'),
	Input('input_table', 'data'),
	Input('scatter_toggle', 'value')
)
def output(rows, toggle):
	global EMPTY
	report = []
	data = []
	for row in rows:
		if (any([cell in EMPTY for cell in row.values()])):
			continue
		if (row['From Year'] > row['To Year']):
			report.append(row_heading(row, year_warning=True))
			report.append(dcc.Graph(figure=go.Figure(
				[],
				layout=go.Layout(
					xaxis={'title': 'Year'},
					yaxis={'title': 'Billing'},
					margin={'r': 20, 'l': 20, 't': 20, 'b': 20}
				)
			)))
			continue
		report.append(row_heading(row))
		stat = row_stat(row)
		plot = row_plot(stat, toggle)
		report.append(row_figure(plot))
		report.append(row_table(stat))
		data.append(row_summary(row, plot))
	summary = summary_figure(data)
	return summary + report

def row_heading(row, **kwargs):
	if 'year_warning' in kwargs and kwargs['year_warning'] == True:
		return html.H1([
				'{}, {} years experience'.format(row['Degree'], row['Experience']),
				html.Br(),
				html.Mark(['From {} to {}'.format(row['From Year'], row['To Year'])], style={
					'background-color': 'red',
					'color': 'yellow'
				}),
				html.Br(),
				'Mean Level: {}, Predict Level: {}'.format(row['Mean Level'], row['Predict Level'])
		])
	else:
		return html.H1([
				'{}, {} years experience'.format(row['Degree'], row['Experience']),
				html.Br(),
				'From {} to {}'.format(row['From Year'], row['To Year']),
				html.Br(),
				'Mean Level: {}, Predict Level: {}'.format(row['Mean Level'], row['Predict Level'])
		])

def row_stat(row):
	global df, model
	year = range(row['From Year'], row['To Year'] + 1)
	line = pd.DataFrame({
		'Year': year,
		'Experience': row['Experience'],
		'Degree': row['Degree']
	})
	X = model_X(line)
	return {
		'year': year,
		'mean': model.mean_interval(X, row['Mean Level']),
		'mean-level': row['Mean Level'],
		'pred': model.predict_interval(X, row['Predict Level']),
		'pred-level': row['Predict Level'],
		'yhat': model.mean(X),
		'data': df[
				(df['Degree'] == row['Degree'])
			& (df['Experience'] == row['Experience'])
			& (df['Year'] >= row['From Year'])
			& (df['Year'] <= row['To Year'])
		]
	}

def row_plot(stat, toggle):
	year = stat['year']
	mean = stat['mean']
	pred = stat['pred']
	yhat = stat['yhat']
	data = stat['data']
	n = len(year)
	plot = {}
	plot['mean'] = go.Figure() if 'mean' not in toggle else go.Figure(
			px.line(x=year, y=mean[:, 0].reshape(n)).data
		+ (() if 'scatter' not in toggle else px.scatter(x=year, y=mean[:, 0].reshape(n)).data)
		+ px.line(x=year, y=mean[:, 1].reshape(n)).data
		+ (() if 'scatter' not in toggle else px.scatter(x=year, y=mean[:, 1].reshape(n)).data)
	)
	plot['pred'] = go.Figure() if 'pred' not in toggle else go.Figure(
			px.line(x=year, y=pred[:, 0].reshape(n)).data
		+ (() if 'scatter' not in toggle else px.scatter(x=year, y=pred[:, 0].reshape(n)).data)
		+ px.line(x=year, y=pred[:, 1].reshape(n)).data
		+ (() if 'scatter' not in toggle else px.scatter(x=year, y=pred[:, 1].reshape(n)).data)
	)
	plot['yhat'] = go.Figure(
			px.line(x=year, y=yhat.reshape(n)).data
		+ (() if 'scatter' not in toggle else px.scatter(x=year, y=yhat.reshape(n)).data)
	)
	plot['data'] = go.Figure() if len(data) == 0 else go.Figure(
		  (() if 'trace' not in toggle else px.line(x=data['Year'], y=data['Billing']).data)
		+ px.scatter(x=data['Year'], y=data['Billing']).data
	)
	scheme = px.colors.qualitative.Plotly
	for p in plot['mean'].data:
		if p['mode'] == 'lines':
			p['line']['color'] = (scheme[0] if 'scatter' not in toggle else scheme[5])
		if p['mode'] == 'markers':
			p['marker']['color'] = scheme[0]
	for p in plot['pred'].data:
		if p['mode'] == 'lines':
			p['line']['color'] = (scheme[1] if 'scatter' not in toggle else scheme[6])
		if p['mode'] == 'markers':
			p['marker']['color'] = scheme[1]
	for p in plot['yhat'].data:
		if p['mode'] == 'lines':
			p['line']['color'] = (scheme[2] if 'scatter' not in toggle else scheme[7])
		if p['mode'] == 'markers':
			p['marker']['color'] = scheme[2]
	for p in plot['data'].data:
		if p['mode'] == 'lines':
			p['line']['color'] = 'gray'
		if p['mode'] == 'markers':
			p['marker']['color'] = 'black'
	for f in plot.values():
		for p in f.data:
			p['hovertemplate'] = 'Year: %{x}<br>Billing: %{y:,.4f}<extra></extra>'
	return plot

def row_figure(plot):
	return dcc.Graph(figure=go.Figure(
		[d for p in plot.values() for d in p.data],
		layout=go.Layout(
			xaxis={'title': 'Year'},
			yaxis={'title': 'Billing'},
			margin={'r': 20, 'l': 20, 't': 20, 'b': 20}
		)
	))

def row_table(stat):
	df = pd.DataFrame(np.c_[stat['year'], stat['yhat'], stat['mean'], stat['pred']])
	df = pd.merge(df, stat['data'][['Year', 'Billing']], left_on=0, right_on='Year', how='left')[
		['Year', 'Billing'] + list(range(1, df.shape[1]))
	]
	df.columns = [
		'Year', 'Inkindo', 'Model',
		'{}% Mean Low'.format(stat['mean-level'] * 100),
		'{}% Mean High'.format(stat['mean-level'] * 100),
		'{}% Predicted Low'.format(stat['pred-level'] * 100),
		'{}% Predicted High'.format(stat['pred-level'] * 100)
	]
	columns = []
	for i in df.columns:
		if i in ['Year', 'Color']:
			columns.append({'name': i, 'id': i})
		elif i == 'Inkindo':
			columns.append({'name': i, 'id': i, 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed, group=Group.yes)})
		else:
			columns.append({'name': i, 'id': i, 'type': 'numeric', 'format': Format(precision=4, scheme=Scheme.fixed, group=Group.yes)})
	return dash_table.DataTable(df.to_dict('records'), columns, cell_selectable=False, style_table={'overflowX': 'scroll'})

def row_summary(row, plot):
	scheme = px.colors.qualitative.Plotly
	data = []
	for p in plot['yhat'].data:
		if p['mode'] == 'lines':
			p['hovertemplate'] = '{}<br>Degree: {}<br>Experience: {}<br>'.format(
				'MODEL', row['Degree'], row['Experience']
			) + 'Year: %{x}<br>Billing: %{y:,.4f}<extra></extra>'
			p['line']['color'] = scheme[row['Color'] % (len(scheme) // 2) + len(scheme) // 2]
			data.append(p)
	for p in plot['data'].data:
		if p['mode'] == 'markers':
			p['hovertemplate'] = '{}<br>Degree: {}<br>Experience: {}<br>'.format(
				'INKINDO', row['Degree'], row['Experience']
			) + 'Year: %{x}<br>Billing: %{y:,.4f}<extra></extra>'
			p['marker']['color'] = scheme[row['Color'] % (len(scheme) // 2)]
			data.append(p)
	return go.Figure(data)

def summary_figure(data):
	if len(data) < 2:
		return []
	return [
		html.H1(['Summary View']),
		dcc.Graph(figure=go.Figure(
			[d for p in data for d in p.data],
			layout=go.Layout(
				xaxis={'title': 'Year'},
				yaxis={'title': 'Billing'},
				margin={'r': 20, 'l': 20, 't': 20, 'b': 20}
			)
		))
	]

def model_X(df):
	global degree
	X = df[['Year', 'Experience']].to_numpy()
	X = np.c_[np.ones(X.shape[0], dtype=X.dtype), X, X[:, 0] * X[:, 1]]
	X = np.hstack([X * (df[['Degree']] == d).astype(int).to_numpy() for d in degree])
	return X

def model_y(df):
	return df[['Billing']].to_numpy()

class linear_regression:
	def __init__(self, X, y):
		self.y = y.copy()
		self.X = X.copy()

		self.k = self.X.shape[1] - 1
		self.v = self.X.shape[0] - self.X.shape[1]
		self.n = self.X.shape[0]

		self.c = inv(self.X.T @ self.X)
		self.b = self.c @ self.X.T @ self.y

		self.yhat = self.mean(self.X)
		self.ybar = np.mean(self.y)

		self.SSR = np.sum((self.yhat - self.ybar) ** 2)
		self.SSE = np.sum((self.y - self.yhat) ** 2)
		self.SST = self.SSR + self.SSE

		self.MSR = self.SSR / self.k
		self.MSE = self.SSE / self.v
		self.MST = self.SST / (self.n - 1)

		self.s2 = self.MSE
		self.s = np.sqrt(self.s2)

		self.f = self.MSR / self.MSE
		self.fp = 1.0 - stats.f.cdf(self.f, self.k, self.v)

		self.R2 = self.SSR / self.SST
		self.R2adj = 1.0 - self.MSE / self.MST

		self.bt = self.b / (self.s * np.sqrt( np.diag(self.c).reshape(self.c.shape[0], 1) ))
		self.btp = 2.0 * stats.t.cdf( -abs(self.bt), self.v )

	def info(self):
		print('b\n', self.b)
		print('n\t', self.n)
		print('k\t', self.k)
		print('v\t', self.v)
		print('ybar\t', self.ybar)
		print('SSR\t', self.SSR)
		print('SSE\t', self.SSE)
		print('SST\t', self.SST)
		print('MSR\t', self.MSR)
		print('MSE\t', self.MSE)
		print('MST\t', self.MST)
		print('s2\t', self.s2)
		print('s\t', self.s)
		print('f\t', self.f)
		print('fp\t', self.fp)
		print('R2\t', self.R2)
		print('R2adj\t', self.R2adj)
		print('bt\n', self.bt)
		print('btp\n', self.btp)

	def mean(self, X):
		return X @ self.b

	def mean_interval(self, X, a):
		return (
			self.mean(X)
			+ stats.t.ppf((1.0 - a) / 2.0, self.v)
			* self.s
			* np.sqrt(np.reshape(
				np.reshape(X, [X.shape[0], 1, X.shape[1]])
				@ self.c
				@ np.reshape(X, [X.shape[0], X.shape[1], 1])
				, [X.shape[0], 1]
			))
			* np.array([[1, -1]])
		)

	def predict_interval(self, X, a):
		return (
			self.mean(X)
			+ stats.t.ppf((1.0 - a) / 2.0, self.v)
			* self.s
			* np.sqrt(np.reshape(
				1.0
				+ np.reshape(X, [X.shape[0], 1, X.shape[1]])
				@ self.c
				@ np.reshape(X, [X.shape[0], X.shape[1], 1])
				, [X.shape[0], 1]
			))
			* np.array([[1, -1]])
		)

if __name__ == '__main__':
	main(__name__, *sys.argv)
