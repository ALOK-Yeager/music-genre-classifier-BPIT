# Fake song recommendations for UI demo purposes
# Maps broad genre groups to lists of famous public-domain or well-known songs

SIMILAR_SONGS = {
    'Rock': [
        {'title': 'House of the Rising Sun', 'artist': 'Traditional (popularized by The Animals)'},
        {'title': 'Greensleeves Rock Version', 'artist': 'Traditional Arrangement'},
        {'title': 'Wild Thing', 'artist': 'The Troggs'}
    ],
    'Pop': [
        {'title': 'Ode to Joy', 'artist': 'Ludwig van Beethoven (Pop Arrangement)'},
        {'title': 'La Cucaracha', 'artist': 'Traditional Mexican Folk'},
        {'title': 'Auld Lang Syne', 'artist': 'Traditional Scottish'}
    ],
    'Classical': [
        {'title': 'Symphony No. 5', 'artist': 'Ludwig van Beethoven'},
        {'title': 'The Four Seasons', 'artist': 'Antonio Vivaldi'},
        {'title': 'Canon in D', 'artist': 'Johann Pachelbel'}
    ],
    'Dance': [
        {'title': 'Blue Danube Waltz', 'artist': 'Johann Strauss II'},
        {'title': 'Hungarian Dance No. 5', 'artist': 'Johannes Brahms'},
        {'title': 'Dance of the Sugar Plum Fairy', 'artist': 'Pyotr Ilyich Tchaikovsky'}
    ],
    'Indie': [
        {'title': 'Scarborough Fair', 'artist': 'Traditional English (Folk Revival)'},
        {'title': 'Wayfaring Stranger', 'artist': 'Traditional American Folk'},
        {'title': 'Shenandoah', 'artist': 'Traditional American'}
    ],
    'Soul': [
        {'title': 'Nobody Knows the Trouble I\'ve Seen', 'artist': 'Traditional Spiritual'},
        {'title': 'Swing Low, Sweet Chariot', 'artist': 'Traditional Spiritual'},
        {'title': 'Sometimes I Feel Like a Motherless Child', 'artist': 'Traditional Spiritual'}
    ]
}


# Mapping from specific genre labels to broad categories
GENRE_TO_BROAD_CATEGORY = {
    # Classical genres
    'Symphony': 'Classical',
    'Opera': 'Classical',
    'Solo': 'Classical',
    'Chamber': 'Classical',
    
    # Pop genres
    'Pop_vocal_ballad': 'Pop',
    'Adult_contemporary': 'Pop',
    'Teen_pop': 'Pop',
    'Acoustic_pop': 'Pop',
    
    # Dance genres
    'Contemporary_dance_pop': 'Dance',
    'Dance_pop': 'Dance',
    
    # Indie genres
    'Classic_indie_pop': 'Indie',
    'Chamber_cabaret_and_art_pop': 'Indie',
    
    # Soul/RnB
    'Soul_or_RnB': 'Soul',
    
    # Rock genres
    'Adult_alternative_rock': 'Rock',
    'Uplifting_anthemic_rock': 'Rock',
    'Soft_rock': 'Rock'
}


def get_recommendations(genre_name):
    """
    Get song recommendations for a given genre.
    
    Parameters:
    -----------
    genre_name : str
        The specific genre name (e.g., 'Symphony', 'Teen_pop')
    
    Returns:
    --------
    list
        List of dictionaries containing song recommendations
    """
    # Map specific genre to broad category
    broad_category = GENRE_TO_BROAD_CATEGORY.get(genre_name, 'Pop')
    
    # Get recommendations for that category
    recommendations = SIMILAR_SONGS.get(broad_category, SIMILAR_SONGS['Pop'])
    
    return recommendations


def format_recommendations(recommendations):
    """
    Format recommendations as a readable string.
    
    Parameters:
    -----------
    recommendations : list
        List of song recommendation dictionaries
    
    Returns:
    --------
    str
        Formatted string of recommendations
    """
    formatted = []
    for i, song in enumerate(recommendations, 1):
        formatted.append(f"{i}. \"{song['title']}\" by {song['artist']}")
    return '\n'.join(formatted)


def get_all_categories():
    """
    Get all broad genre categories.
    
    Returns:
    --------
    list
        List of all broad category names
    """
    return list(SIMILAR_SONGS.keys())


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("FAKE SONG RECOMMENDATIONS (For UI Demo Only)")
    print("=" * 70)
    
    # Show all categories
    print("\nAvailable broad genre categories:")
    for category in get_all_categories():
        print(f"  - {category}")
    
    # Show recommendations for each category
    print("\n" + "=" * 70)
    for category, songs in SIMILAR_SONGS.items():
        print(f"\n{category} Recommendations:")
        print("-" * 70)
        for i, song in enumerate(songs, 1):
            print(f"  {i}. \"{song['title']}\"")
            print(f"     Artist: {song['artist']}")
    
    # Test the recommendation function
    print("\n" + "=" * 70)
    print("\nTest: Getting recommendations for specific genres:")
    print("-" * 70)
    
    test_genres = ['Symphony', 'Teen_pop', 'Dance_pop', 'Adult_alternative_rock']
    for genre in test_genres:
        print(f"\nGenre: {genre}")
        broad_cat = GENRE_TO_BROAD_CATEGORY.get(genre, 'Pop')
        print(f"Broad Category: {broad_cat}")
        recommendations = get_recommendations(genre)
        print(format_recommendations(recommendations))
    
    print("\n" + "=" * 70)
    print("Note: These are public-domain songs for demonstration purposes only.")
    print("=" * 70)
