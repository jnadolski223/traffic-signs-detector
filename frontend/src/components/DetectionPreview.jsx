export default function DetectionPreview({ imageUrl }) {
    return (
        <div style={styles.wrapper}>
            <h2 style={styles.sectionTitle}>Detection result</h2>
            <img src={imageUrl} alt="Analyzed image" style={styles.analyzedImage} />
        </div>
    );
};

const styles = {
    wrapper: {
        width: "50%",
        height: "100%",
        margin: "1rem",
        border: "5px solid #007bff",
        borderRadius: "10px",
        padding: "1rem",
        backgroundColor: "#eee"
    },
    analyzedImage: {
        width: "100%",
        maxHeight: "500px",
        objectFit: "contain",
        // border: "2px solid #444"
    },
    sectionTitle: {
        textAlign: "center",
        marginBottom: "1rem",
        color: "#444"
    }
};