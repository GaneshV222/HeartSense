export default function PageContainer({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      flex: 1,
      padding: '2rem',
      maxWidth: '1200px',
      margin: '0 auto',
      width: '100%'
    }}>
      {children}
    </div>
  );
}
