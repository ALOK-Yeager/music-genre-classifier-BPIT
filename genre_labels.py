# Genre label mapping for ccmusic-database/music_genre dataset
# Maps integer class IDs (0-15) to human-readable music genre names
# Based on thr_level_label from the dataset

GENRE_LABELS = {
    0: "Symphony",
    1: "Opera",
    2: "Solo",
    3: "Chamber",
    4: "Pop_vocal_ballad",
    5: "Adult_contemporary",
    6: "Teen_pop",
    7: "Contemporary_dance_pop",
    8: "Dance_pop",
    9: "Classic_indie_pop",
    10: "Chamber_cabaret_and_art_pop",
    11: "Soul_or_RnB",
    12: "Adult_alternative_rock",
    13: "Uplifting_anthemic_rock",
    14: "Soft_rock",
    15: "Acoustic_pop"
}

# Reverse mapping: genre name to class ID
ID_TO_GENRE = GENRE_LABELS
GENRE_TO_ID = {genre: idx for idx, genre in GENRE_LABELS.items()}

# Number of classes
NUM_CLASSES = len(GENRE_LABELS)


def get_genre_name(class_id):
    """
    Get the genre name for a given class ID.
    
    Parameters:
    -----------
    class_id : int
        The class ID (0-15)
    
    Returns:
    --------
    str
        The corresponding genre name
    """
    return GENRE_LABELS.get(class_id, "Unknown")


def get_class_id(genre_name):
    """
    Get the class ID for a given genre name.
    
    Parameters:
    -----------
    genre_name : str
        The genre name
    
    Returns:
    --------
    int or None
        The corresponding class ID, or None if not found
    """
    return GENRE_TO_ID.get(genre_name, None)


def get_all_genres():
    """
    Get a list of all genre names.
    
    Returns:
    --------
    list
        List of all genre names in order
    """
    return [GENRE_LABELS[i] for i in range(NUM_CLASSES)]


# Example usage
if __name__ == "__main__":
    print("Music Genre Classification Labels")
    print("=" * 50)
    print(f"\nTotal number of classes: {NUM_CLASSES}\n")
    
    print("Class ID -> Genre Name:")
    for class_id, genre_name in GENRE_LABELS.items():
        print(f"  {class_id:2d}: {genre_name}")
    
    print("\n" + "=" * 50)
    
    # Test the helper functions
    print("\nExample usage:")
    print(f"Genre for class ID 0: {get_genre_name(0)}")
    print(f"Genre for class ID 15: {get_genre_name(15)}")
    print(f"Class ID for 'Symphony': {get_class_id('Symphony')}")
    print(f"Class ID for 'Acoustic_pop': {get_class_id('Acoustic_pop')}")
    
    print("\nAll genres:")
    for genre in get_all_genres():
        print(f"  - {genre}")
