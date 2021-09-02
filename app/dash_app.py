import dash
# from jupyter_dash import JupyterDash
import dash_bio as dashbio
# from dash_table import DataTable
# from dash_bio_utils import protein_reader
# import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from appHelperFuncs import *
from zoonosisHelperFunctions import createHostNewickTree

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

app.layout = \
dbc.Container([
    html.Center([
        html.Div([
            html.H3(id='project-heading', 
                    children='Investigating Protein Sequences to Elucidate Species Crossover Events in Viral Infectious Disease', style={'textAlign':'center'}),
            html.H4(id='project-sub-heading', children='Visualisation of Virus-Host Data', style={'textAlign':'center'})
        ], id='titles'),
    ]),
    html.Div([
        dbc.Row([
            dbc.Col([
                # Table button
                dbc.Button()
            ]),
            dbc.Col([
                # Checklist
                dcc.Checklist(# check list to include either of the visualizations
                    id='check-list-visualizations',
                    options=[ # list of label-value dictionary pairs
                        {'label': 'Data Table', 'value': 'virus-host-dataframe'},
                        {'label': 'Network Graph', 'value': 'virus-host-compound-nodes'},
                        {'label': 'Phylo Tree', 'value': 'hosts-phylogenetic-tree'},
                        {'label': 'Alignment Chart', 'value': 'proteins-alignment-chart'},
                    ],
                    value=['virus-host-compound-nodes', 'hosts-phylogenetic-tree', 'proteins-alignment-chart',], # default values
                    labelStyle={'display': 'inline-block'}, # style as single line checklist
                )
            ])
        ]),
        dbc.Row([
            dbc.Col([
                # Dropdowns
                dcc.Dropdown(# proteins drop-down menu
                    id='protein-drop-down',
                    options=[# list of label-value dictionary pairs
                        {'label': record['Protein'], 
                         'value': record['Protein']} for record in df.to_dict('records')
                    ],
                    value='Spike glycoprotein',
                    clearable=True,
                    searchable=True
                ),
            ]),
            dbc.Col([
                dcc.Dropdown(# virus drop-down menu
                    id='virus-drop-down',
                    options=[# list of label-value dictionary pairs
                        {'label': record['Species name'], 
                         'value': record['Species name'],} for record in df.to_dict('records')
                    ],
                    value='Human coronavirus',
                    clearable=True,
                    searchable=True,
                    multi=True,
                ), 
            ])
        ]),
    ], id='selectors'),
    
    dbc.Spinner([
        table
    ], color='primary', type='grow'),

    html.Div(
        id='no-table-viz',
        children=[
            visualizations
        ]),
    html.Div(
    id='proteins-alignment-chart-div', 
    children=[# Alignment chart of similar named proteins
        dashbio.AlignmentChart(
            id='proteins-alignment-chart', # have to read proteins from fasta file
#             data=data ######### read in
        )
    ])
])

f'virus has {2} protein(s) involved in viral entry to host cell'

tree_df = (df.applymap(str).groupby('Species name', as_index=False)
           .agg({'Virus host name':', '.join,'Virus hosts ID':', '.join}))

tree_df['Host Newick data'] = (tree_df['Virus hosts ID']
                               .swifter.progress_bar(desc='Creating newick tree objects')
                               .apply(createHostNewickTree))

@app.callback(
    Output(component_id='virus-host-compound-nodes', component_property='elements'),
    Input(component_id='virus-drop-down', component_property='value')
)
def update_compund_nodes(input_value):
    parentNodes = [{'data': {'id': 'virus', 'label': 'Virus'}}, 
                   {'data': {'id': 'host', 'label': 'Host(s)'}},]

    dff = df[df['Species name'].isin(input_value)]
    
    virusesHosts = dff[['Virus host name', 'Species name']].to_dict('records')
    uniqe_virus_records = dff[['Species name']].drop_duplicates().to_dict('records')
    uniqe_host_records = dff[['Virus host name']].drop_duplicates().to_dict('records')
    
    virusNames = [{'data': {'id': record['Species name'].lower(), 
                            'label': record['Species name'], 'parent': 'virus',
                            'position': {'x': 100, 'y': (n+1)*100}}} for n, record in enumerate(uniqe_virus_records)]
    
    virusHosts = [{'data': {'id': record['Virus host name'].lower(), 
                            'label': record['Virus host name'],'parent': 'host',
                            'position': {'x': 400, 'y': (n+1)*100}}} for n, record in enumerate(uniqe_host_records)]
    
    childrenNodes = virusNames + virusHosts
    
    nodeEdges = [{'data': {'source': record['Species name'], 
                           'target': record['Virus host name'],
                           'classes': 'virus-host-edge'}} for record in virusesHosts]
    
    elements=parentNodes+childrenNodes+nodeEdges
    return elements

# uniqe_protein_records = dff[['Protein names']].drop_duplicates().to_dict('records')

@app.callback(
    Output(component_id='proteins-alignment-chart', component_property='data'),
    Input(component_id='protein-drop-down', component_property='value')
)
def update_alignment_chart(selection):
    with open(f'/home/rserage/Documents/Masters Project/Data/Proteins-alignment/{selection}.fasta', 'r') as f:
        alignment = f.read()
    return alignment

@app.callback(
    Output(component_id='hosts-phylogenetic-tree', component_property='elements'),
    Input(component_id='phylo-dropdown', component_property='value')
)
def update_phylogenetic_tree(input_value):
    tree = tree_df[tree_df['Species name'] == input_value]['Host Newick data']
    nodes, edges = generate_elements(tree)
    elements = nodes + edges
    return elements


@app.callback(Output('hosts-phylogenetic-tree', 'stylesheet'),
              Input('hosts-phylogenetic-tree', 'mouseoverEdgeData')
)
def color_children(edgeData):
    if not edgeData:
        return stylesheet

    if 's' in edgeData['source']:
        val = edgeData['source'].split('s')[0]
    else:
        val = edgeData['source']

    children_style = [{
        'selector': 'edge[source *= "{}"]'.format(val),
        'style': {
            'line-color': 'blue'
        }
    }]

    return stylesheet + children_style

if __name__ == '__main__':
    app.run_server(debug=True)#mode='external', host="localhost")