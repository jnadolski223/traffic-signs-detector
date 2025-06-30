import RoadSignInfo from "./RoadSignInfo"

export default function ClassificationPreview({ roadSigns }) {
    return (
        <div style={styles.wrapper}>
            <h2 style={styles.sectionTitle}>Classified road signs</h2>
            {roadSigns.length > 0 ? roadSigns.map((sign, idx) => <RoadSignInfo key={idx} roadSign={sign}/>) : <p>No signs detected</p>}
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
    sectionTitle: {
        textAlign: "center",
        marginBottom: "1rem",
        color: "#444"
    }
};