const IMAGE_EXTENSION = ".png";

export default function RoadSignInfo({ roadSign }) {
    return (
        <div style={styles.wrapper}>
            <img src={`/${roadSign.class_name}${IMAGE_EXTENSION}`} alt={`Road sign ${roadSign.class_name}`} style={styles.image} />
            <div style={styles.description}>
                <p><strong>Sign name: </strong> {roadSign.class_name}</p>
                <p><strong>Probability: </strong> {Number(roadSign.confidence) * 100}%</p>
                <p><strong>Bounding box: </strong> [{roadSign.bbox.join(", ")}]</p>
            </div>
        </div>
    );
};

const styles = {
    wrapper: {
        margin: "1rem",
        border: "5px solid #ccc",
        borderRadius: "10px",
        padding: "1rem",
        display: "flex",
        justifyContent: "space-evenly",
        alignItems: "center",
        backgroundColor: "#fee"
    },
    image: {
        width: "20%"
    },
    description: {
        textAlign: "left"
    }
};