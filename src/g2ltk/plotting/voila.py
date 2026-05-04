import os

import matplotlib.pyplot as plt
from IPython.display import HTML, display
from . import activate_saveplot

def is_voila_active():
    are_we_in_voila = os.environ.get('SERVER_SOFTWARE', '').startswith('voila')
    # are_we_in_voila = ('SERVER_SOFTWARE' in os.environ and 'voila' in os.environ['SERVER_SOFTWARE']) or 'QUERY_STRING' in os.environ # also works
    return are_we_in_voila

def apply_css(latex=False, font_size=18):
    activate_saveplot(latex, font_size=font_size, style='prez')
    plt.rcParams['figure.constrained_layout.use'] = False
    plt.rcParams['figure.autolayout'] = True
    if not is_voila_active():
        HTML_string = """
        <style>
            /* 1. HIDE BUTTONS */
            .jupyter-matplotlib-toolbar button[title="Download plot"],
            .jupyter-matplotlib-toolbar button[title="Back to previous view"],
            .jupyter-matplotlib-toolbar button[title="Forward to next view"] {
                display: none !important;
            }
        </style>
        """
    else:
        HTML_string = """<style>
    /* 1. HIDE BUTTONS & STYLE TOOLBAR */
    .jupyter-matplotlib-toolbar button[title="Download plot"],
    .jupyter-matplotlib-toolbar button[title="Back to previous view"],
    .jupyter-matplotlib-toolbar button[title="Forward to next view"] {
        display: none !important;
    }

    .jupyter-matplotlib-toolbar {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
        background: transparent !important;
        border: none !important;
    }

    /* 2. RESPONSIVE MAGIC */
    .jupyter-widgets.jupyter-matplotlib {
        width: 100% !important; 
        max-width: 100%;
        height: auto !important; 
        margin: 0 auto !important;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    canvas.jupyter-matplotlib-canvas {
        width: 100% !important;
        max-width: fit-content;
        height: auto !important;
    }

    /* 3. LAYOUT & OUTPUT AREA CENTERING */
    .jp-Cell-outputArea, 
    .jp-OutputArea-child, 
    .jp-OutputArea-output,
    .jp-RenderedHTMLCommon {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        width: 100% !important;
        text-align: center !important;
    }
    
    /* 4. MARKDOWN & TEXT CENTERING */
    /* Target the specific container for Markdown text */
    .rendered_html, 
    .jp-RenderedMarkdown, 
    .jp-RenderedHTMLCommon {
        font-size: 18pt !important;
        text-align: center !important;
        width: 100% !important;
    }

    /* Ensure all block elements inside are centered */
    .rendered_html p, 
    .rendered_html ul, 
    .rendered_html ol,
    .jp-RenderedHTMLCommon p,
    .jp-RenderedHTMLCommon ul,
    .jp-RenderedHTMLCommon ol {
        text-align: center !important;
        list-style-position: inside !important; /* Keeps bullet points centered with text */
        width: 100%;
    }

    /* Headings */
    .rendered_html h1, .jp-RenderedHTMLCommon h1 { font-size: 2em !important; }
    .rendered_html h2, .jp-RenderedHTMLCommon h2 { font-size: 1.6em !important; }
    .rendered_html h3, .jp-RenderedHTMLCommon h3 { font-size: 1.3em !important; }

    /* MathJax / KaTeX */
    .MathJax, .MathJax_Display, .katex, .katex-display {
        font-size: 1.2em !important;
        text-align: center !important;
        margin: 0 auto !important;
    }
    
    /* 5. REMOVE FIGURE LABELS (Widget Headers) */
    .jupyter-matplotlib-header,
    .jupyter-matplotlib-title {
        display: none !important;
    }
</style>"""
        # HTML_string ="""
        # <style>
        #     /* 1. HIDE BUTTONS */
        #     .jupyter-matplotlib-toolbar button[title="Download plot"],
        #     .jupyter-matplotlib-toolbar button[title="Back to previous view"],
        #     .jupyter-matplotlib-toolbar button[title="Forward to next view"] {
        #         display: none !important;
        #     }
        #
        #     .jupyter-matplotlib-toolbar {
        #         display: flex !important;
        #         justify-content: center !important;
        #         width: 100% !important;
        #         background: transparent !important;
        #         border: none !important;
        #     }
        #
        #     /* 2. RESPONSIVE MAGIC: Make the widget and canvas fluid */
        #     .jupyter-widgets.jupyter-matplotlib {
        #         width: 100% !important; /* Allow the widget container to fill space */
        #         max-width: 100%;        /* But never overflow its parent */
        #         height: auto !important;
        #         margin: 0 auto !important;
        #         display: flex;
        #         flex-direction: column;
        #         align-items: center;
        #     }
        #
        #     canvas.jupyter-matplotlib-canvas {
        #         width: 100% !important;   /* Let the canvas scale down */
        #         max-width: fit-content;   /* But don't let it scale UP larger than your figsize */
        #         height: auto !important;  /* Maintain aspect ratio */
        #     }
        #
        #     /* 3. LAYOUT CENTERING */
        #     .jp-Cell-outputArea,
        #     .jp-OutputArea-child,
        #     .jp-OutputArea-output {
        #         display: flex !important;
        #         flex-direction: column !important;
        #         align-items: center !important;
        #         width: 100% !important;
        #     }
        #
        #     /* === NEW: MARKDOWN FONT SIZE & CENTERING === */
        #
        #     /* Base font size for all markdown cells */
        #     .jp-MarkdownCell .jp-RenderedMarkdown,
        #     .jp-MarkdownCell .jp-RenderedHTMLCommon {
        #         font-size: 18pt !important;
        #         text-align: center !important;
        #     }
        #
        #     /* Headings scale relative to the new base */
        #     .jp-MarkdownCell .jp-RenderedHTMLCommon h1 { font-size: 2em !important; }
        #     .jp-MarkdownCell .jp-RenderedHTMLCommon h2 { font-size: 1.6em !important; }
        #     .jp-MarkdownCell .jp-RenderedHTMLCommon h3 { font-size: 1.3em !important; }
        #
        #     /* MathJax / KaTeX equations */
        #     .jp-MarkdownCell .MathJax,
        #     .jp-MarkdownCell .MathJax_Display,
        #     .jp-MarkdownCell .katex,
        #     .jp-MarkdownCell .katex-display {
        #         font-size: 1.2em !important;   /* relative to the 18px base above */
        #       //  text-align: center !important;
        #       //  display: block !important;
        #     }
        #
        #     /* Center block-level elements inside markdown */
        #     // remove this if the block test do not need to be centered
        #     .jp-MarkdownCell .jp-RenderedHTMLCommon p,
        #     .jp-MarkdownCell .jp-RenderedHTMLCommon ul,
        #     .jp-MarkdownCell .jp-RenderedHTMLCommon ol {
        #         text-align: center !important;
        #     }
        #
        #     .jp-MarkdownCell .jp-RenderedHTMLCommon {
        #         text-align: center !important;
        #     }
        #
        #     </style>
        #     <script>
        #         window.addEventListener('message', function(e) {
        #             if (e.data && e.data.type === 'voila-scroll') {
        #                 window.scrollBy({top: e.data.amount, behavior: 'smooth'});
        #             }
        #         });
        #     </script>
        #     """
    display(HTML(HTML_string))

import uuid

def end_slide(label="Next ↓"):
    if not is_voila_active(): return
    uid = uuid.uuid4().hex[:8]
    display(HTML(f"""
    <div id="before-{uid}"></div>
    <div style='position:sticky; bottom:0; display:flex; justify-content:center; 
                padding: 16px 0; background: linear-gradient(transparent, white 40%);'>
        <button onclick="
            var el = document.querySelector('.lm-Widget.lm-BoxPanel-child');
            var btn = this.closest('div[style*=sticky]');
            var btnBottom = btn.getBoundingClientRect().bottom;
            var elTop = el.getBoundingClientRect().top;
            var after = document.getElementById('after-{uid}');
            var target = el.scrollTop + (btnBottom - elTop) + after.clientHeight;
            el.scrollTo({{top: target, behavior: 'smooth'}});
        "
                style='padding: 12px 32px; font-size: 18px; cursor: pointer;
                       border: 1.5px solid #aaa; border-radius: 8px; 
                       background: white; color: #333;'>
            {label}
        </button>
    </div>
    <div id="after-{uid}"></div>
    <script>
        setTimeout(function() {{
            var before = document.getElementById('before-{uid}');
            var after  = document.getElementById('after-{uid}');
            var el = document.querySelector('.lm-Widget.lm-BoxPanel-child');
            var alreadyUsed = before.getBoundingClientRect().top - el.getBoundingClientRect().top;
            var needed = el.clientHeight - alreadyUsed - 70;
            before.style.height = Math.max(0, needed) + 'px';
            after.style.height = el.clientHeight + 'px';
        }}, 300);
    </script>
    """))