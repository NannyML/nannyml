/*
 * # Author:   Niels Nuyttens  <niels@nannyml.com>
 * #
 * # License: Apache Software License 2.0
 */

window.dataLayer = window.dataLayer || [];
  function gtag(){window.dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-BRPYB8Q3DC', {'send_page_view': false });


jQuery(document).on('cut copy', function(){
    const currentPage = jQuery(document).attr('title');
    let words = []
    let selection = window.getSelection() + '';
    words = selection.split(' ')
    const wordsLength = words.length;

    let copy_event = {
        page: currentPage,
        words: words,
    };
    console.log('logging copy event: ' + copy_event);
    gtag("event", "docs_copy", JSON.stringify(copy_event));
});
