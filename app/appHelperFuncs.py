import math
import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_table import DataTable
import dash_cytoscape as cyto

def generate_elements(tree, xlen=30, ylen=30, grabbable=False):
    def get_col_positions(tree, column_width=80):
        """Create a mapping of each clade to its column position."""
        taxa = tree.get_terminals()

        # Some constants for the drawing calculations
        max_label_width = max(len(str(taxon)) for taxon in taxa)
        drawing_width = column_width - max_label_width - 1

        depths = tree.depths()
        # If there are no branch lengths, assume unit branch lengths
        if not max(depths.values()):
            depths = tree.depths(unit_branch_lengths=True)
            # Potential drawing overflow due to rounding -- 1 char per tree layer
        fudge_margin = int(math.ceil(math.log(len(taxa), 2)))
        cols_per_branch_unit = ((drawing_width - fudge_margin) /
                                float(max(depths.values())))
        return dict((clade, int(blen * cols_per_branch_unit + 1.0))
                    for clade, blen in depths.items())

    def get_row_positions(tree):
        taxa = tree.get_terminals()
        positions = dict((taxon, 2 * idx) for idx, taxon in enumerate(taxa))

        def calc_row(clade):
            for subclade in clade:
                if subclade not in positions:
                    calc_row(subclade)
            positions[clade] = ((positions[clade.clades[0]] +
                                 positions[clade.clades[-1]]) // 2)

        calc_row(tree.root)
        return positions

    def add_to_elements(clade, clade_id):
        children = clade.clades

        pos_x = col_positions[clade] * xlen
        pos_y = row_positions[clade] * ylen

        cy_source = {
            "data": {"id": clade_id},
            'position': {'x': pos_x, 'y': pos_y},
            'classes': 'nonterminal',
            'grabbable': grabbable
        }
        nodes.append(cy_source)

        if clade.is_terminal():
            cy_source['data']['name'] = clade.name
            cy_source['classes'] = 'terminal'

        for n, child in enumerate(children):
            """The "support" node is on the same column as the parent clade,
            and on the same row as the child clade. It is used to create the
            90 degree angle between the parent and the children.
            Edge config: parent -> support -> child"""

            support_id = clade_id + 's' + str(n)
            child_id = clade_id + 'c' + str(n)
            pos_y_child = row_positions[child] * ylen

            cy_support_node = {
                'data': {'id': support_id},
                'position': {'x': pos_x, 'y': pos_y_child},
                'grabbable': grabbable,
                'classes': 'support'
            }

            cy_support_edge = {
                'data': {
                    'source': clade_id,
                    'target': support_id,
                    'sourceCladeId': clade_id
                },
            }

            cy_edge = {
                'data': {
                    'source': support_id,
                    'target': child_id,
                    'length': clade.branch_length,
                    'sourceCladeId': clade_id
                },
            }

            if clade.confidence: #and clade.confidence.value:
                cy_source['data']['confidence'] = clade.confidence #.value

            nodes.append(cy_support_node)
            edges.extend([cy_support_edge, cy_edge])

            add_to_elements(child, child_id)

    col_positions = get_col_positions(tree)
    row_positions = get_row_positions(tree)

    nodes = []
    edges = []

    add_to_elements(tree.clade, 'r')

    return nodes, edges




stylesheet = [
    {
        'selector': '.nonterminal',
        'style': {
            'label': 'data(confidence)',
            'background-opacity': 0,
            "text-halign": "left",
            "text-valign": "top",
        }
    },
    {
        'selector': '.support',
        'style': {'background-opacity': 0}
    },
    {
        'selector': 'edge',
        'style': {
            "source-endpoint": "inside-to-node",
            "target-endpoint": "inside-to-node",
        }
    },
    {
        'selector': '.terminal',
        'style': {
            'label': 'data(name)',
            'width': 10,
            'height': 10,
            "text-valign": "center",
            "text-halign": "right",
            'background-color': '#222222'
        }
    }
]

df = pd.read_csv()

table = html.Div(
    id='virus-host-dataframe-div',
    children=[
        dbc.Row(
            dbc.Col([
                DataTable(
                    id='virus-host-dataframe',
                    columns=[{"name": i, "id": i} for i in df.columns.tolist()],
                    data=df.to_dict('records'),
                    style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
                    style_table={'height': '400px', 'overflowY': 'auto', 'overflowX':'auto'},
                    style_cell={'textAlign':'left'},
                    filter_action="native",
                    sort_action="native",
                    page_action="native",
                    page_current= 0,
                    page_size= 100,
                    column_selectable="single",
                    row_selectable="single",
                    selected_columns=[],
                    selected_rows=[],)
                    ]))
                    ])



network_map = html.Div(
    id='virus-host-compound-nodes-div',
    children=[# Virus-host cytoscape as compound nodes
    cyto.Cytoscape(
        id='virus-host-compound-nodes',
        layout={'name': 'preset'},
        style={'width': '100%', 'height': '450px'},
        stylesheet=[
            {'selector': 'node','style': {'content': 'data(label)'}},
            {'selector': '.virus-host-edge','style': {'line-style': 'dashed'}}
        ])
    ])


phylo_drpdwn = dbc.Row(
    dbc.Col([
        dcc.Dropdown(
            id='phylo-dropdown',
            options=[# list of label-value dictionary pairs
                {'label': record['Species name'], 
                    'value': record['Species name']} for record in df.to_dict('records')
            ],
            value='',
            clearable=True,
            searchable=True,
            )
        ])),
        

phylo_tree = dbc.Row([
    dbc.Col(
    id='hosts-phylogenetic-tree-div',
    children=[
    # Phylogenetic tree of hosts
    cyto.Cytoscape(
        id='hosts-phylogenetic-tree', # have to read phylogeny from ete3 xml file
#                                     elements=elements, # define elements from tree
#                                     stylesheet=stylesheet, # define style sheet
        layout={'name': 'preset'},
        style={'height': '95vh', 'width': '100%'}
    )
])], width=5)
# ])

visualizations = dbc.Row([
    # Data visualisations
    dbc.Col([
        dbc.Spinner([
            network_map
        ]),
    ], width=5),
    dbc.Col([
        phylo_drpdwn,phylo_tree
    ], justify='around')])