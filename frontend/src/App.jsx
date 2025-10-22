import { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import DetectionPreview from "./components/DetectionPreview";
import ClassificationPreview from "./components/ClassificationPreview";

export default function App() {
    const [analyzedImageUrl, setAnalyzedImageUrl] = useState(null);
    const [detectedSigns, setDetectedSigns] = useState(null);

    const handleEvaluate = async (file) => {
        const formData = new FormData();
        formData.append("image", file);
        try {
            const response = await fetch("http://localhost:5000/analyze", { method: "POST", body: formData });
            if (!response.ok) throw new Error("Błąd podczas przesyłania obrazu.");
            const result = await response.json();
            setDetectedSigns(result.detected_signs || []);
            setAnalyzedImageUrl(`data:image/jpeg;base64,${result.image_base64}`);
        } catch (err) {
            console.error("Błąd:", err.message);
        };
    };

    const handleReset = () => {
        setAnalyzedImageUrl(null);
        setDetectedSigns(null);
    };

    return (
        <div style={styles.wrapper}>
            <h1 style={styles.header}>Road Sign Detector</h1>
            <ImageUploader onEvaluate={handleEvaluate} onReset={handleReset} />
            <div style={styles.resultsContainer}>
                {analyzedImageUrl && <DetectionPreview imageUrl={analyzedImageUrl} />}
                {detectedSigns && <ClassificationPreview  roadSigns={detectedSigns} />}
            </div>
        </div>
    );
};

const styles = {
    wrapper: {
        fontFamily: "Arial, sans-serif"
    },
    header: {
        padding: "15px",
        textAlign: "center",
        width: "100%",
        backgroundColor: "darkblue",
        color: "#fff"
    },
    resultsContainer: {
        display: "flex",
        alignItems: "stretch",
        justifyContent: "space-evenly"
    }
};