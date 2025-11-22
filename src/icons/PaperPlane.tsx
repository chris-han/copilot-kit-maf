import { Send as LucideSend, LucideProps } from 'lucide-react';

const PaperPlane = ({ className, ...props }: LucideProps) => {
  return <LucideSend className={className} {...props} />;
};

export default PaperPlane;