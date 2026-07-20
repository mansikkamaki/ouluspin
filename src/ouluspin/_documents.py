# SPDX-FileCopyrightText: 2020-2026 Akseli Mansikkamäki
# SPDX-License-Identifier: GPL-3.0-or-later

"""Internal writers of the word-processor renderings of a result table.

The module writes the OpenDocument text (.odt) and the Office Open XML
(.docx) renderings of the tables of the result_table module. Both formats
are ZIP archives of XML documents, so the writers are built on the zipfile
module of the standard library and the library needs no external
dependencies for them.

The module is internal to the library and is not part of the public API.
It is imported by the odt_table and docx_table methods of ResultTable when
they are called, so that the ordinary use of the library never loads it.

The tables are laid out in the style used in the scientific literature:
the columns are separated by white space only, a horizontal rule is drawn
above and below the header of the table and below its last row, and the
table spans the full width of the text area of an A4 page.

The renderers take the content of the table as the dictionary returned by
the private __document_content method of ResultTable, which contains the
following items:

    title        : the title of the table, or None
    notes        : the list of the lines printed above the table
    summary      : the list of the lines printed below the table
    footnotes    : the list of the footnotes, without their markers
    header_lines : the list of the lines of the column header, each a list
                   of the headers of the columns
    body         : the list of the rows of the table (see below)
    alignments   : the list of the alignments of the columns
    widths       : the list of the plain-text widths of the columns, used
                   to divide the width of the table between the columns
    n_columns    : the number of columns of the table

Each item of the body is a dictionary. An ordinary row has the type 'row'
and contains the list of its cells as (text,is_number) pairs together with
the flag rule_above telling whether a horizontal rule is drawn above the
row. A section heading has the type 'section' and contains its text.
"""

import os
import shutil
import zipfile


# The dimensions of the page. The tables are laid out for an A4 page with
# margins of 2 cm, i.e. for a text area of 6.6875 in, which is the width
# the tables of the example document temp/table_example.odt were written
# for.
PAGE_WIDTH    = 8.2677
PAGE_HEIGHT   = 11.6929
PAGE_MARGIN   = 0.7874
TEXT_WIDTH    = PAGE_WIDTH - 2.0*PAGE_MARGIN

# The font of the tables. The size is the one the rows were dimensioned
# for; a larger size may overflow the width of the page.
FONT_NAME     = "Liberation Serif"
FONT_SIZE     = 11.0

# The thickness of the horizontal rules of the tables.
RULE_WIDTH    = "0.5pt"

# The typographic minus sign. The negative numbers of the documents are
# written with it instead of the hyphen of the plain-text table.
MINUS_SIGN    = "−"

# The height the subscripts and the superscripts are raised or lowered by,
# and their size, as a percentage of the height of the font.
SCRIPT_OFFSET = "58%"
SCRIPT_SIZE   = "58%"


def escape(text):
    """Return the text with the characters that are special in XML
    replaced by the corresponding entities.
    """
    return str(text).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


def column_widths(content):
    """Return the widths of the columns of the table in inches. The text
    area of the page is divided between the columns in proportion to the
    widths the columns have in the plain-text rendering, so that the
    columns of long entries are given more room than the narrow ones.
    """
    widths = [max(1,width) for width in content['widths']]
    total  = float(sum(widths))

    return [TEXT_WIDTH*width/total for width in widths]


def plain_segments(text):
    """Return a text as a single unformatted segment, in the segment form
    the renderers build their cells and paragraphs of (see the
    markup_segments class method of ResultTable).
    """
    return [{'text': str(text), 'italic': False, 'position': 'normal'}]


def number_segments(text):
    """Return a formatted number as a single segment, with the hyphens of
    the plain-text form replaced by the typographic minus sign.
    """
    return [{'text':     str(text).replace("-",MINUS_SIGN),
             'italic':   False,
             'position': 'normal'}]


def header_segments(text):
    """Return a column or row header as a list of formatted segments.

    The physical quantities of the header are recognized and set with
    their subscripts and superscripts, and a footnote marker at the end of
    the header is set as an italic superscript instead of the 'a)' of the
    plain-text table.
    """
    from ouluspin.result_table import ResultTable

    body, marker = ResultTable.split_footnote_marker(text)

    segment_list = ResultTable.markup_segments(body)

    if marker is not None:
        segment_list.append({'text':     marker,
                             'italic':   True,
                             'position': 'super'})

    return segment_list


def cell_segments(item, column, row_header_column=False):
    """Return one cell of a row of the table as a list of formatted
    segments. The numbers are set with the typographic minus sign and the
    cells of the row header column are typeset as the headers they are.
    """
    if column >= len(item['cells']):
        return []

    text, is_number = item['cells'][column]

    if is_number:
        return number_segments(text)

    if row_header_column and column == 0:
        return header_segments(text)

    return plain_segments(text)


def footnote_segment_lines(content):
    """Return the footnotes of the table as a list of the paragraphs they
    are printed as, one per footnote, each given as a list of formatted
    segments. The marker of a footnote is set as an italic superscript.
    """
    line_list = []

    for i in range(0,len(content['footnotes'])):
        line_list.append([{'text':     chr(ord('a') + i),
                           'italic':   True,
                           'position': 'super'},
                          {'text':     " " + joined_text(content['footnotes'][i]),
                           'italic':   False,
                           'position': 'normal'}])

    return line_list


def joined_text(text):
    """Return a note, a summary line or a footnote as a single line.

    An entry of several lines is wrapped by hand for the plain-text table,
    which cannot wrap it itself. A word-processor document wraps the text
    of a paragraph on its own, so the manual line breaks are dropped and
    the lines are joined back into one.
    """
    return " ".join([line.strip() for line in str(text).split("\n")
                     if not line.strip() == ""])


def summary_lines(content):
    """Return the summary of the table as a list of the paragraphs it is
    printed as, one per entry of the summary.
    """
    return [joined_text(text) for text in content['summary']]


def note_lines(content):
    """Return the notes of the table as a list of the paragraphs they are
    printed as, one per note.
    """
    return [joined_text(text) for text in content['notes']]


def zip_document(filename, file_list):
    """Write a ZIP archive containing the given files.

    Arguments
    ---------
    filename : str
        The name of the archive to write.
    file_list : list
        The files of the archive as (name,content,stored) tuples, where
        the content is a str and the flag stored tells whether the entry
        is stored without compression, as the mimetype entry of an
        OpenDocument file must be.
    """
    archive = zipfile.ZipFile(filename,'w',zipfile.ZIP_DEFLATED)

    for name, file_content, stored in file_list:
        if stored:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info,file_content)
        else:
            archive.writestr(name,file_content)

    archive.close()


def replace_in_archive(filename, replacements):
    """Rewrite the given entries of an existing ZIP archive, keeping all
    the other entries as they are. The archive is rewritten into a
    temporary file which replaces the original one only after it has been
    written successfully, so that a failure cannot destroy the document
    that was appended to.

    Arguments
    ---------
    filename : str
        The name of the archive to modify.
    replacements : dict
        The new contents of the entries to rewrite, keyed by the names of
        the entries.
    """
    temporary_name = filename + ".ouluspin_tmp"

    source = zipfile.ZipFile(filename,'r')
    target = zipfile.ZipFile(temporary_name,'w',zipfile.ZIP_DEFLATED)

    try:
        for item in source.infolist():
            data = source.read(item.filename)

            if item.filename in replacements:
                data = replacements[item.filename].encode('utf-8')

            # The mimetype entry of an OpenDocument file must be stored
            # without compression.
            if item.compress_type == zipfile.ZIP_STORED:
                info = zipfile.ZipInfo(item.filename,date_time=item.date_time)
                info.compress_type = zipfile.ZIP_STORED
                target.writestr(info,data)
            else:
                target.writestr(item.filename,data)
    finally:
        source.close()
        target.close()

    shutil.move(temporary_name,filename)


def insert_before(document, marker, addition):
    """Return the document with the addition inserted before the last
    occurrence of the marker. The document is returned unchanged when it
    does not contain the marker.
    """
    position = document.rfind(marker)
    if position < 0:
        return document

    return document[:position] + addition + document[position:]


#
# The OpenDocument text (.odt) renderer.
#

ODT_NAMESPACES = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
    'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"'
)


def odt_styles_document():
    """Return the styles.xml of a new OpenDocument text document. The
    document defines the A4 page layout and the paragraph styles used by
    the tables.
    """
    text_properties = ('<style:text-properties style:font-name="' + FONT_NAME
                       + '" fo:font-size="' + str(FONT_SIZE) + 'pt"/>')

    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<office:document-styles ' + ODT_NAMESPACES + ' office:version="1.3">'
            '<office:styles>'
            '<style:default-style style:family="paragraph">'
            + text_properties +
            '</style:default-style>'
            '<style:style style:name="Standard" style:family="paragraph">'
            + text_properties +
            '</style:style>'
            '<style:style style:name="Table_20_Contents"'
            ' style:display-name="Table Contents" style:family="paragraph"'
            ' style:parent-style-name="Standard">'
            '<style:paragraph-properties fo:margin-top="0in" fo:margin-bottom="0in"/>'
            + text_properties +
            '</style:style>'
            '<style:style style:name="Table_20_Heading"'
            ' style:display-name="Table Heading" style:family="paragraph"'
            ' style:parent-style-name="Table_20_Contents">'
            '<style:paragraph-properties fo:margin-top="0in" fo:margin-bottom="0in"/>'
            '<style:text-properties style:font-name="' + FONT_NAME
            + '" fo:font-size="' + str(FONT_SIZE) + 'pt" fo:font-weight="bold"/>'
            '</style:style>'
            '</office:styles>'
            '<office:automatic-styles>'
            '<style:page-layout style:name="PageLayout">'
            '<style:page-layout-properties'
            ' fo:page-width="' + "{0:.4f}".format(PAGE_WIDTH) + 'in"'
            ' fo:page-height="' + "{0:.4f}".format(PAGE_HEIGHT) + 'in"'
            ' style:print-orientation="portrait"'
            ' fo:margin-top="' + "{0:.4f}".format(PAGE_MARGIN) + 'in"'
            ' fo:margin-bottom="' + "{0:.4f}".format(PAGE_MARGIN) + 'in"'
            ' fo:margin-left="' + "{0:.4f}".format(PAGE_MARGIN) + 'in"'
            ' fo:margin-right="' + "{0:.4f}".format(PAGE_MARGIN) + 'in"/>'
            '</style:page-layout>'
            '</office:automatic-styles>'
            '<office:master-styles>'
            '<style:master-page style:name="Standard"'
            ' style:page-layout-name="PageLayout"/>'
            '</office:master-styles>'
            '</office:document-styles>')


def odt_manifest():
    """Return the META-INF/manifest.xml of a new OpenDocument text
    document.
    """
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<manifest:manifest'
            ' xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"'
            ' manifest:version="1.3">'
            '<manifest:file-entry manifest:full-path="/"'
            ' manifest:media-type="application/vnd.oasis.opendocument.text"/>'
            '<manifest:file-entry manifest:full-path="content.xml"'
            ' manifest:media-type="text/xml"/>'
            '<manifest:file-entry manifest:full-path="styles.xml"'
            ' manifest:media-type="text/xml"/>'
            '</manifest:manifest>')


def odt_table_styles(content, prefix):
    """Return the automatic styles of one table of an OpenDocument text
    document as a string. The names of the styles are prefixed by the
    given prefix, which makes them unique within a document containing
    several tables.
    """
    width_list = column_widths(content)

    tmp_str = ('<style:style style:name="' + prefix + '" style:family="table">'
               '<style:table-properties style:width="'
               + "{0:.4f}".format(TEXT_WIDTH) + 'in" table:align="left"/>'
               '</style:style>')

    for i in range(0,len(width_list)):
        tmp_str += ('<style:style style:name="' + prefix + '.' + str(i)
                    + '" style:family="table-column">'
                    '<style:table-column-properties style:column-width="'
                    + "{0:.4f}".format(width_list[i]) + 'in"/>'
                    '</style:style>')

    # The cell styles. The header cells carry a rule above and below, the
    # cells of the last row carry a rule below, the cells of a row
    # following a rule row carry a rule above, and the ordinary cells
    # carry no rules at all.
    cell_style_list = (('header',RULE_WIDTH,RULE_WIDTH),
                       ('body',None,None),
                       ('above',RULE_WIDTH,None),
                       ('last',None,RULE_WIDTH))

    for name, top, bottom in cell_style_list:
        if top is None:
            top_str = 'fo:border-top="none"'
        else:
            top_str = 'fo:border-top="' + top + ' solid #000000"'

        if bottom is None:
            bottom_str = 'fo:border-bottom="none"'
        else:
            bottom_str = 'fo:border-bottom="' + bottom + ' solid #000000"'

        tmp_str += ('<style:style style:name="' + prefix + '.' + name
                    + '" style:family="table-cell">'
                    '<style:table-cell-properties style:vertical-align="middle"'
                    ' fo:padding="0.0382in" fo:border-left="none"'
                    ' fo:border-right="none" ' + top_str + ' ' + bottom_str + '/>'
                    '</style:style>')

    # The paragraph styles of the cells, one per alignment.
    for alignment, name in (('l','left'),('r','right'),('c','center')):
        for parent, style in (('Table_20_Contents','contents'),
                              ('Table_20_Heading','heading')):
            tmp_str += ('<style:style style:name="' + prefix + '.' + style + '.'
                        + name + '" style:family="paragraph"'
                        ' style:parent-style-name="' + parent + '">'
                        '<style:paragraph-properties fo:text-align="'
                        + name + '" style:justify-single-word="false"/>'
                        '</style:style>')

    # The text styles of the formatted pieces of the cells, i.e. the
    # italics of the symbols and the subscripts and the superscripts of
    # the indices carried by them.
    for style_name, properties in text_style_properties():
        tmp_str += ('<style:style style:name="' + prefix + '.' + style_name
                    + '" style:family="text">'
                    '<style:text-properties ' + properties + '/>'
                    '</style:style>')

    return tmp_str


def text_style_properties():
    """Return the text styles the formatted pieces of the cells are set
    with as (name,properties) pairs, one per combination of the italics
    and the vertical position of a segment.
    """
    style_list = []

    for position, position_str in (('normal',""),
                                   ('sub','style:text-position="sub '
                                    + SCRIPT_OFFSET + ' ' + SCRIPT_SIZE + '"'),
                                   ('super','style:text-position="super '
                                    + SCRIPT_OFFSET + ' ' + SCRIPT_SIZE + '"')):
        for italic, italic_str in ((False,""),
                                   (True,'fo:font-style="italic"')):
            if position == 'normal' and not italic:
                continue

            properties = " ".join([text for text in (position_str,italic_str)
                                   if not text == ""])

            style_list.append((text_style_name(italic,position),properties))

    return style_list


def text_style_name(italic, position):
    """Return the name of the text style of a segment, without the prefix
    of the table, or None when the segment needs no style at all.
    """
    if position == 'normal' and not italic:
        return None

    if italic:
        return position + "_italic"

    return position


def odt_segments(segment_list, prefix):
    """Return the formatted pieces of a cell or of a paragraph of an
    OpenDocument text document, the formatted ones being wrapped in the
    text styles of the table.
    """
    tmp_str = ""

    for segment in segment_list:
        style_name = text_style_name(segment['italic'],segment['position'])

        if style_name is None:
            tmp_str += escape(segment['text'])
        else:
            tmp_str += ('<text:span text:style-name="' + prefix + '.'
                        + style_name + '">' + escape(segment['text'])
                        + '</text:span>')

    return tmp_str


def odt_cell(segment_list, cell_style, paragraph_style, prefix):
    """Return one table cell of an OpenDocument text document."""
    return ('<table:table-cell table:style-name="' + cell_style
            + '" office:value-type="string">'
            '<text:p text:style-name="' + paragraph_style + '">'
            + odt_segments(segment_list,prefix) +
            '</text:p></table:table-cell>')


def odt_paragraph(segment_list, prefix, style="Standard"):
    """Return one paragraph of an OpenDocument text document. The text of
    the paragraph is given as a list of formatted segments.
    """
    return ('<text:p text:style-name="' + style + '">'
            + odt_segments(segment_list,prefix) + '</text:p>')


def odt_table_body(content, prefix):
    """Return the body of one table of an OpenDocument text document, i.e.
    the title, the notes, the table itself, the summary and the footnotes.
    """
    alignments = content['alignments']
    n_columns  = content['n_columns']

    def paragraph_style(column, heading):
        if heading:
            style = 'heading'
        else:
            style = 'contents'

        if column < len(alignments) and alignments[column] == 'l':
            name = 'left'
        elif column < len(alignments) and alignments[column] == 'c':
            name = 'center'
        else:
            name = 'right'

        return prefix + '.' + style + '.' + name

    tmp_str = ""

    if content['title'] is not None:
        tmp_str += odt_paragraph(plain_segments(content['title']),prefix)

    for line in note_lines(content):
        tmp_str += odt_paragraph(plain_segments(line),prefix)

    tmp_str += ('<table:table table:name="' + prefix
                + '" table:style-name="' + prefix + '">')

    for i in range(0,n_columns):
        tmp_str += ('<table:table-column table:style-name="' + prefix + '.'
                    + str(i) + '"/>')

    # The header lines. The rules above and below the header are drawn on
    # the cells of the header rows.
    header_lines = content['header_lines']
    for i in range(0,len(header_lines)):
        if len(header_lines) == 1:
            cell_style = prefix + '.header'
        elif i == 0:
            cell_style = prefix + '.above'
        elif i == len(header_lines) - 1:
            cell_style = prefix + '.last'
        else:
            cell_style = prefix + '.body'

        tmp_str += '<table:table-row>'
        for j in range(0,n_columns):
            if j < len(header_lines[i]):
                segment_list = header_segments(header_lines[i][j])
            else:
                segment_list = []
            tmp_str += odt_cell(segment_list,cell_style,
                                paragraph_style(j,True),prefix)
        tmp_str += '</table:table-row>'

    # The body of the table. The rule below the table is drawn on the
    # cells of the last row.
    row_list = [item for item in content['body'] if item['type'] == 'row'
                or item['type'] == 'section']

    for i in range(0,len(row_list)):
        item = row_list[i]
        last = (i == len(row_list) - 1)

        if last and item.get('rule_above',False):
            cell_style = prefix + '.header'
        elif last:
            cell_style = prefix + '.last'
        elif item.get('rule_above',False):
            cell_style = prefix + '.above'
        else:
            cell_style = prefix + '.body'

        tmp_str += '<table:table-row>'

        if item['type'] == 'section':
            # A section heading spans the whole width of the table.
            tmp_str += ('<table:table-cell table:style-name="' + cell_style
                        + '" table:number-columns-spanned="' + str(n_columns)
                        + '" office:value-type="string">'
                        '<text:p text:style-name="' + prefix + '.heading.left">'
                        + escape(item['text']) +
                        '</text:p></table:table-cell>')
            for j in range(1,n_columns):
                tmp_str += '<table:covered-table-cell/>'
        else:
            for j in range(0,n_columns):
                tmp_str += odt_cell(cell_segments(item,j,
                                                  content['has_row_headers']),
                                    cell_style,paragraph_style(j,False),prefix)

        tmp_str += '</table:table-row>'

    tmp_str += '</table:table>'

    for line in summary_lines(content):
        tmp_str += odt_paragraph(plain_segments(line),prefix)

    for segment_list in footnote_segment_lines(content):
        tmp_str += odt_paragraph(segment_list,prefix)

    # An empty paragraph separates the table from what follows it.
    tmp_str += odt_paragraph(plain_segments(""),prefix)

    return tmp_str


def next_table_index(document, prefix):
    """Return the index of the next table of a document, i.e. one more
    than the largest index already used by the automatic styles of the
    tables written by the library. The indices keep the style names of the
    tables of one document unique.
    """
    import re

    index = 0
    for match in re.finditer(re.escape(prefix) + r'(\d+)',document):
        index = max(index,int(match.group(1)) + 1)

    return index


def write_odt(content, filename):
    """Write the table into an OpenDocument text (.odt) file, appending it
    to the document when the file already exists and creating the document
    when it does not.

    Arguments
    ---------
    content : dict
        The content of the table (see the module docstring).
    filename : str
        The name of the file to write.
    """
    if os.path.exists(filename):
        archive  = zipfile.ZipFile(filename,'r')
        document = archive.read('content.xml').decode('utf-8')
        archive.close()

        prefix = 'OuluSpinTable' + str(next_table_index(document,'OuluSpinTable'))

        document = insert_before(document,'</office:automatic-styles>',
                                 odt_table_styles(content,prefix))
        document = insert_before(document,'</office:text>',
                                 odt_table_body(content,prefix))

        replace_in_archive(filename,{'content.xml': document})
        return

    prefix = 'OuluSpinTable0'

    document = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<office:document-content ' + ODT_NAMESPACES
                + ' office:version="1.3">'
                '<office:automatic-styles>'
                + odt_table_styles(content,prefix) +
                '</office:automatic-styles>'
                '<office:body><office:text>'
                + odt_table_body(content,prefix) +
                '</office:text></office:body>'
                '</office:document-content>')

    zip_document(filename,
                 [('mimetype','application/vnd.oasis.opendocument.text',True),
                  ('META-INF/manifest.xml',odt_manifest(),False),
                  ('styles.xml',odt_styles_document(),False),
                  ('content.xml',document,False)])


#
# The Office Open XML (.docx) renderer.
#

DOCX_NAMESPACES = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"')


def docx_content_types():
    """Return the [Content_Types].xml of a new .docx document."""
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels"'
            ' ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml"'
            ' ContentType="application/vnd.openxmlformats-officedocument'
            '.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml"'
            ' ContentType="application/vnd.openxmlformats-officedocument'
            '.wordprocessingml.styles+xml"/>'
            '</Types>')


def docx_relationships():
    """Return the _rels/.rels of a new .docx document."""
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships'
            ' xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships'
            '/officeDocument" Target="word/document.xml"/>'
            '</Relationships>')


def docx_document_relationships():
    """Return the word/_rels/document.xml.rels of a new .docx document."""
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships'
            ' xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships'
            '/styles" Target="styles.xml"/>'
            '</Relationships>')


def docx_styles_document():
    """Return the word/styles.xml of a new .docx document, defining the
    font of the document.
    """
    # The font size is given in half-points in the .docx format.
    half_points = str(int(round(2.0*FONT_SIZE)))

    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:styles ' + DOCX_NAMESPACES + '>'
            '<w:docDefaults><w:rPrDefault><w:rPr>'
            '<w:rFonts w:ascii="' + FONT_NAME + '" w:hAnsi="' + FONT_NAME + '"/>'
            '<w:sz w:val="' + half_points + '"/>'
            '</w:rPr></w:rPrDefault>'
            '<w:pPrDefault><w:pPr>'
            '<w:spacing w:after="0" w:line="240" w:lineRule="auto"/>'
            '</w:pPr></w:pPrDefault>'
            '</w:docDefaults>'
            '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
            '<w:name w:val="Normal"/>'
            '</w:style>'
            '</w:styles>')


def docx_section_properties():
    """Return the section properties of a .docx document, i.e. the A4 page
    size and the margins. The dimensions are given in twentieths of a
    point.
    """
    def twips(inches):
        return str(int(round(1440.0*inches)))

    return ('<w:sectPr>'
            '<w:pgSz w:w="' + twips(PAGE_WIDTH) + '" w:h="' + twips(PAGE_HEIGHT) + '"/>'
            '<w:pgMar w:top="' + twips(PAGE_MARGIN) + '"'
            ' w:right="' + twips(PAGE_MARGIN) + '"'
            ' w:bottom="' + twips(PAGE_MARGIN) + '"'
            ' w:left="' + twips(PAGE_MARGIN) + '"'
            ' w:header="0" w:footer="0" w:gutter="0"/>'
            '</w:sectPr>')


def docx_runs(segment_list, bold=False):
    """Return the formatted pieces of a paragraph of a .docx document as
    the runs of the paragraph.
    """
    tmp_str = ""

    for segment in segment_list:
        properties = ""

        if bold:
            properties += '<w:b/>'
        if segment['italic']:
            properties += '<w:i/>'
        if segment['position'] == 'sub':
            properties += '<w:vertAlign w:val="subscript"/>'
        elif segment['position'] == 'super':
            properties += '<w:vertAlign w:val="superscript"/>'

        if not properties == "":
            properties = '<w:rPr>' + properties + '</w:rPr>'

        tmp_str += ('<w:r>' + properties + '<w:t xml:space="preserve">'
                    + escape(segment['text']) + '</w:t></w:r>')

    return tmp_str


def docx_paragraph(segment_list, bold=False, alignment='l'):
    """Return one paragraph of a .docx document. The text of the paragraph
    is given as a list of formatted segments.
    """
    if alignment == 'r':
        justification = 'right'
    elif alignment == 'c':
        justification = 'center'
    else:
        justification = 'left'

    return ('<w:p><w:pPr><w:jc w:val="' + justification + '"/></w:pPr>'
            + docx_runs(segment_list,bold) + '</w:p>')


def docx_cell(segment_list, width, top_rule, bottom_rule, bold=False,
              alignment='l', span=1):
    """Return one table cell of a .docx document. The rules of the cell are
    drawn as its top and bottom borders.
    """
    if top_rule:
        top_str = '<w:top w:val="single" w:sz="4" w:color="000000"/>'
    else:
        top_str = '<w:top w:val="none" w:sz="0" w:color="auto"/>'

    if bottom_rule:
        bottom_str = '<w:bottom w:val="single" w:sz="4" w:color="000000"/>'
    else:
        bottom_str = '<w:bottom w:val="none" w:sz="0" w:color="auto"/>'

    if span > 1:
        span_str = '<w:gridSpan w:val="' + str(span) + '"/>'
    else:
        span_str = ''

    return ('<w:tc><w:tcPr>'
            '<w:tcW w:w="' + str(int(round(1440.0*width))) + '" w:type="dxa"/>'
            + span_str +
            '<w:tcBorders>'
            + top_str +
            '<w:left w:val="none" w:sz="0" w:color="auto"/>'
            + bottom_str +
            '<w:right w:val="none" w:sz="0" w:color="auto"/>'
            '</w:tcBorders>'
            '<w:vAlign w:val="center"/>'
            '</w:tcPr>'
            + docx_paragraph(segment_list,bold,alignment) +
            '</w:tc>')


def docx_table_body(content):
    """Return the body of one table of a .docx document, i.e. the title,
    the notes, the table itself, the summary and the footnotes.
    """
    width_list = column_widths(content)
    alignments = content['alignments']
    n_columns  = content['n_columns']

    def alignment_of(column):
        if column < len(alignments):
            return alignments[column]
        return 'r'

    def cell_width(column):
        if column < len(width_list):
            return width_list[column]
        return TEXT_WIDTH/max(1,n_columns)

    tmp_str = ""

    if content['title'] is not None:
        tmp_str += docx_paragraph(plain_segments(content['title']))

    for line in note_lines(content):
        tmp_str += docx_paragraph(plain_segments(line))

    tmp_str += ('<w:tbl><w:tblPr>'
                '<w:tblW w:w="' + str(int(round(1440.0*TEXT_WIDTH)))
                + '" w:type="dxa"/>'
                '<w:tblLayout w:type="fixed"/>'
                '<w:tblCellMar>'
                '<w:top w:w="55" w:type="dxa"/><w:bottom w:w="55" w:type="dxa"/>'
                '<w:left w:w="55" w:type="dxa"/><w:right w:w="55" w:type="dxa"/>'
                '</w:tblCellMar>'
                '</w:tblPr><w:tblGrid>')

    for i in range(0,n_columns):
        tmp_str += '<w:gridCol w:w="' + str(int(round(1440.0*cell_width(i)))) + '"/>'

    tmp_str += '</w:tblGrid>'

    header_lines = content['header_lines']
    for i in range(0,len(header_lines)):
        top_rule    = (i == 0)
        bottom_rule = (i == len(header_lines) - 1)

        tmp_str += '<w:tr>'
        for j in range(0,n_columns):
            if j < len(header_lines[i]):
                segment_list = header_segments(header_lines[i][j])
            else:
                segment_list = []
            tmp_str += docx_cell(segment_list,cell_width(j),top_rule,
                                 bottom_rule,True,alignment_of(j))
        tmp_str += '</w:tr>'

    row_list = [item for item in content['body'] if item['type'] == 'row'
                or item['type'] == 'section']

    for i in range(0,len(row_list)):
        item        = row_list[i]
        top_rule    = item.get('rule_above',False)
        bottom_rule = (i == len(row_list) - 1)

        tmp_str += '<w:tr>'

        if item['type'] == 'section':
            tmp_str += docx_cell(plain_segments(item['text']),TEXT_WIDTH,
                                 top_rule,bottom_rule,True,'l',n_columns)
        else:
            for j in range(0,n_columns):
                tmp_str += docx_cell(cell_segments(item,j,
                                                   content['has_row_headers']),
                                     cell_width(j),top_rule,bottom_rule,
                                     False,alignment_of(j))

        tmp_str += '</w:tr>'

    tmp_str += '</w:tbl>'

    for line in summary_lines(content):
        tmp_str += docx_paragraph(plain_segments(line))

    for segment_list in footnote_segment_lines(content):
        tmp_str += docx_paragraph(segment_list)

    tmp_str += docx_paragraph(plain_segments(""))

    return tmp_str


def write_docx(content, filename):
    """Write the table into an Office Open XML (.docx) file, appending it
    to the document when the file already exists and creating the document
    when it does not.

    Arguments
    ---------
    content : dict
        The content of the table (see the module docstring).
    filename : str
        The name of the file to write.
    """
    if os.path.exists(filename):
        archive  = zipfile.ZipFile(filename,'r')
        document = archive.read('word/document.xml').decode('utf-8')
        archive.close()

        # The section properties, when present, must remain the last
        # element of the body, so the table is inserted before them.
        if document.rfind('<w:sectPr') >= 0:
            document = insert_before(document,'<w:sectPr',docx_table_body(content))
        else:
            document = insert_before(document,'</w:body>',docx_table_body(content))

        replace_in_archive(filename,{'word/document.xml': document})
        return

    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<w:document ' + DOCX_NAMESPACES + '><w:body>'
                + docx_table_body(content)
                + docx_section_properties() +
                '</w:body></w:document>')

    zip_document(filename,
                 [('[Content_Types].xml',docx_content_types(),False),
                  ('_rels/.rels',docx_relationships(),False),
                  ('word/_rels/document.xml.rels',docx_document_relationships(),False),
                  ('word/styles.xml',docx_styles_document(),False),
                  ('word/document.xml',document,False)])
